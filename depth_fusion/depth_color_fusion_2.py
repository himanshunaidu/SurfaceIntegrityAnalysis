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
depth_max    = 200.0        # far clip in meters

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
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'broken_sidewalk_2'))
FRAMES_PATH = os.path.join(DATASET_PATH, 'dataset.csv')
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
last_frustum_block_coords = None
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
    last_frustum_block_coords = frustum_block_coords
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
pcd_legacy = pcd.to_legacy()
# o3d.visualization.draw([pcd])

# mesh = vbg.extract_triangle_mesh()
# o3d.visualization.draw([mesh.to_legacy()])

# Save point cloud
# o3d.io.write_point_cloud("output.ply", pcd_legacy)
# exit(-1)

# --- 3) Raycast a depth map from the nth camera ---
n      = len(frames) - 1  # target frame index
K_n    = frames[n]["K"].to(device)
# Twc_n  = o3d.core.Tensor(frames[n]["T_w_c"].astype(np.float32), device=device)
# Tc_w_n = o3d.core.linalg.inv(Twc_n)              # camera <- world
Twc_n = frames[n]["T_w_c"].astype(np.float64)
# Tc_w_n = np.linalg.inv(Twc_n)  # camera <- world
Tc_w_n = Twc_n.copy()

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

# MARK: Old way to raycast
# rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
#     intrinsic_matrix=K_n,  # Use the intrinsic matrix directly
#     extrinsic_matrix=Tc_w_n,  # world <- camera
#     width_px=W, height_px=H
# )
# ans   = scene.cast_rays(rays)
# depth = ans["t_hit"].reshape((H, W)).cpu().numpy()
# depth[np.isinf(depth)] = 0.0   # mark misses invalid
print(type(Tc_w_n))
# Tc_w_n = Tc_w_n
result = vbg.ray_cast(
    block_coords=last_frustum_block_coords,
    intrinsic=K_n,
    extrinsic=Tc_w_n,
    width=W, height=H,
    render_attributes=[
        'depth', 'normal', 'color', 'index',
        'interp_ratio'
    ],
    depth_scale=depth_scale,
    depth_min=0,
    depth_max=depth_max,
    weight_threshold=1,
    range_map_down_factor=0
)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2)
# Colorized depth
colorized_depth = o3d.t.geometry.Image(result['depth']).colorize_depth(
    depth_scale, 0, depth_max)
axs[0, 0].imshow(colorized_depth.as_tensor().cpu().numpy())
axs[0, 0].set_title('depth')

axs[0, 1].imshow(result['normal'].cpu().numpy())
axs[0, 1].set_title('normal')

axs[1, 0].imshow(result['color'].cpu().numpy())
axs[1, 0].set_title('color via kernel')

plt.show()