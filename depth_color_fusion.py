import os
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image
import cv2
from utils.read_dataset_fusion import get_intrinsics, get_pose_matrix, get_depth, get_depth_confidence, get_rgb

# --- Config (tune for your scene) ---
voxel_size   = 3.0 / 512      # meters per voxel
sdf_trunc    = 5 * voxel_size
block_res    = 8          # voxels per block (default)
device       = o3d.core.Device("CPU:0")  # or "CUDA:0"
depth_scale  = 1000.0    # meters per depth unit; set to your data
depth_max    = 20.0        # far clip in meters

IMG_WIDTH = 1920
IMG_HEIGHT = 1440
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
scale_x = DEPTH_WIDTH / IMG_WIDTH
scale_y = DEPTH_HEIGHT / IMG_HEIGHT

# frames: list of dicts with:
#   depth  -> o3d.t.geometry.Image (float32 meters or uint16_t raw; set scale)
#   K      -> o3d.core.Tensor [[fx,0,cx],[0,fy,cy],[0,0,1]] (float64)
#   T_w_c  -> 4x4 world<-camera (float64)
#   conf   -> optional numpy/torch array HxW in [0,1] (same resolution as depth)
DATASET_PATH = 'dataset/broken_sidewalk_2/'
FRAMES_PATH = 'dataset/broken_sidewalk_2/dataset.csv'
frames_df = pd.read_csv(FRAMES_PATH)
frames = []
for index, row in frames_df.iterrows():
    color_image = get_rgb(row, DATASET_PATH, width=DEPTH_WIDTH, height=DEPTH_HEIGHT, rotation_code=cv2.ROTATE_90_COUNTERCLOCKWISE)
    depth_image = get_depth(row, DATASET_PATH, width=DEPTH_WIDTH, height=DEPTH_HEIGHT, rotation_code=cv2.ROTATE_90_COUNTERCLOCKWISE)
    K = get_intrinsics(row, scale_x=scale_x, scale_y=scale_y)
    T_w_c = get_pose_matrix(row)
    conf = get_depth_confidence(row, DATASET_PATH, width=DEPTH_WIDTH, height=DEPTH_HEIGHT, rotation_code=cv2.ROTATE_90_COUNTERCLOCKWISE) if 'depth_confidence_frame_path' in row else None
    frames.append({
        "rgb": color_image,
        "depth": depth_image,
        "K": o3d.core.Tensor(K, dtype=o3d.core.Dtype.Float64, device=device),  # Ensure K is on the correct device
        "T_w_c": T_w_c.astype(np.float64),
        "conf": conf
    })

# --- 1) Build a VoxelBlockGrid (TSDF + weight, optional color) ---
vbg = o3d.t.geometry.VoxelBlockGrid(
    attr_names=["tsdf", "weight", "color"],                         # add "color" if needed
    attr_dtypes=[o3d.core.Dtype.Float32, o3d.core.Dtype.Float32, o3d.core.Dtype.Float32],
    attr_channels=[1, 1, 3],
    voxel_size=voxel_size,
    block_resolution=block_res,
    block_count=1000000,     # capacity hint; increase for larger scenes
    device=device
)

# Helper: apply confidence as a mask (simple and effective)
# TSDF ignores invalid depth (0.0), so we can use it to mask out low-confidence pixels.
def mask_depth_by_conf(depth_img_t, conf_np, thr=1):
    d = depth_img_t.as_tensor().cpu().numpy().copy()
    d[conf_np < thr] = 0.0                # 0 = invalid in Open3D
    return o3d.t.geometry.Image(o3d.core.Tensor(d, device=device))

# --- 2) Integrate each frame ---
for f in frames:
    rgb = f["rgb"].to(device)
    depth = f["depth"].to(device)
    K     = f["K"].to(device)                    # 3x3
    Twc   = f["T_w_c"].astype(np.float64)
    extr  = o3d.core.Tensor(Twc, device=device)  # world <- camera

    if "conf" in f and f["conf"] is not None:
        depth = mask_depth_by_conf(depth, f["conf"], thr=0)

    # Activate blocks visible from this frame
    frustum_block_coords = vbg.compute_unique_block_coordinates(
        depth, K, extr, depth_scale, depth_max
    )
    # exit(-1)

    # Integrate TSDF (depth only)
    vbg.integrate(
        frustum_block_coords,
        depth,                  # depth image
        rgb,                    # color image
        K,                      # depth intrinsics
        K,                      # color intrinsics (same as depth)
        extr,                   # world <- camera
        depth_scale,
        depth_max
    )

sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
sphere.paint_uniform_color([0.1, 0.1, 0.7]) # Blue color

pcd = vbg.extract_point_cloud()
o3d.visualization.draw([pcd, sphere])
exit(-1)

# --- 3) Raycast a depth map from the nth camera ---
n      = 0 # len(frames) - 1  # target frame index
K_n    = frames[n]["K"].to(device)
# Twc_n  = o3d.core.Tensor(frames[n]["T_w_c"].astype(np.float32), device=device)
# Tc_w_n = o3d.core.linalg.inv(Twc_n)              # camera <- world
Twc_n = frames[n]["T_w_c"].astype(np.float32)
Tc_w_n = np.linalg.inv(Twc_n)  # camera <- world

Tc_w_n_cache = Tc_w_n.copy()

Twc_n = o3d.core.Tensor(Twc_n, device=device)
Tc_w_n = o3d.core.Tensor(Tc_w_n, device=device)

H, W = int(DEPTH_HEIGHT), int(DEPTH_WIDTH)
# fx, fy = float(K_n[0,0].cpu().numpy()), float(K_n[1,1].cpu().numpy())
# cx, cy = float(K_n[0,2].cpu().numpy()), float(K_n[1,2].cpu().numpy())

# Extract mesh or directly raycast the implicit surface. Mesh works across versions:
mesh = vbg.extract_triangle_mesh()
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)

rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    intrinsic_matrix=K_n,  # Use the intrinsic matrix directly
    extrinsic_matrix=Tc_w_n,  # world <- camera
    width_px=W, height_px=H
)
ans   = scene.cast_rays(rays)
depth = ans["t_hit"].reshape((H, W)).cpu().numpy()
depth[np.isinf(depth)] = 0.0   # mark misses invalid

### Debugging
# 3) Choose a sparse grid of pixels to visualize (e.g., every 40 px)
step = 10
vv, uu = np.mgrid[0:H:step, 0:W:step]
idx = (vv * W + uu).ravel()

# Rays come as shape (H, W, 6) or (H*W, 6); reshape to (-1,6) safely
rays_np = rays.numpy().reshape(-1, 6)
O = rays_np[idx, 0:3]   # origins in world coords
D = rays_np[idx, 3:6]   # directions (unit) in world coords
t_hit = ans["t_hit"].numpy()
t = t_hit.reshape(-1)[idx]

# 4) Endpoints: hit -> O + t*D ; miss -> O + L*D
L = 0.25  # 25 cm for visualization
finite = np.isfinite(t) & (t > 0)
E = np.where(finite[:, None], O + D * t[:, None], O + D * L)

# 5) Build a LineSet with per-line colors (green = hit, red = miss)
points = np.vstack([O, E])
lines  = np.column_stack([np.arange(len(O)), np.arange(len(O), 2*len(O))])
colors = np.tile([[0,1,0]], (lines.shape[0], 1))
colors[~finite] = [1, 0, 0]

ls = o3d.geometry.LineSet()
ls.points = o3d.utility.Vector3dVector(points)
ls.lines  = o3d.utility.Vector2iVector(lines)
ls.colors = o3d.utility.Vector3dVector(colors)

# 6) Also draw the camera frame at this view (needs world<-camera)
# If you have T_w_c_n already, use it; otherwise invert T_c_w_n:
T_w_c_n = Tc_w_n_cache # np.linalg.inv(np.asarray(Tc_w_n, dtype=np.float64))
cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
cam_frame.transform(T_w_c_n)

o3d.visualization.draw([mesh, ls, cam_frame])
exit(-1)

### Debugging end

print(f"Raycasted depth map shape: {depth.shape}")
print(f"Depth values range: {np.min(depth)} to {np.max(depth)}")

# 'depth' is your detailed, fused depth map for frame n (meters).
# Save depth as an image if needed:
depth_image = Image.fromarray((depth * 1000).astype(np.uint16))
depth_image = depth_image.rotate(-90, expand=True)  # Rotate the image 90 degrees clockwise
depth_image.save(os.path.join(DATASET_PATH, f"fused_depth_{n}.png"))