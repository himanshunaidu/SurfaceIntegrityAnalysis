import os
import numpy as np
import open3d as o3d
from skimage import morphology, measure

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.plane import get_plane_mesh, get_viz_with_transparency

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset", "lab_controlled", "experiment_6"))
PCD_PATH = os.path.join(DATASET_PATH, "pcd_cropped", "0-0-3-1-c.ply")
pcd_original = o3d.io.read_point_cloud(PCD_PATH)

# Downsample
pcd = pcd_original.voxel_down_sample(voxel_size=0.01)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])

# Detect boundary points
# radius and max_nn: Hybrid nearest search parameters
# angle_threshold: Maximum angle between normals to be considered a boundary point
pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd)
boundary_pcd, boundary_mask = pcd_t.compute_boundary_points(radius=0.1, max_nn=30, angle_threshold=45.0)

# Create a new point cloud without the detected boundary points, using the mask
pcd = pcd.select_by_index(np.where(~boundary_mask.numpy())[0])
# o3d.visualization.draw_geometries([pcd])
# exit(-1)

# o3d.visualization.draw_geometries([pcd])
# Remove statistical outliers (tune nb_neighbors/std_ratio for your density)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

# Recompute normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(50)

# Fit plane
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
plane_mesh = get_plane_mesh(pcd, plane_model, inliers, side_length=5)
# o3d.visualization.draw_geometries([pcd_original, plane_mesh])
# exit(-1)
ground = pcd.select_by_index(inliers)
non_ground = pcd.select_by_index(inliers, invert=True)

# Flatten to 2D by aligning to Z-axis
[a, b, c, d] = plane_model
plane_normal = np.array([a, b, c], dtype=float)
# print("Plane Normal:", plane_normal)
z = np.array([0, 0, 1])
v = np.cross(plane_normal, z)
s = np.linalg.norm(v)
c = float(np.dot(plane_normal, z))
R = np.eye(3)
## If plane_normal is not completely aligned with z-axis
if s > 1e-9:
    # print("Rotating point cloud to align with Z-axis")
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

pcd_rot = o3d.geometry.PointCloud()
pcd_rot.points = o3d.utility.Vector3dVector(pcd.points @ R.T)
pcd_rot.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
plane_rot_model, inliers_rot = pcd_rot.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
plane_rot_mesh = get_plane_mesh(pcd_rot, plane_rot_model, inliers_rot, side_length=5)
# o3d.visualization.draw_geometries([pcd_rot])
# exit(-1)

pcd_rot_points = np.asarray(pcd_rot.points)
pcd_rot_xy = pcd_rot_points[:, :2]
pcd_rot_z = pcd_rot_points[:, 2]
pcd_rot_colors = np.asarray(pcd_rot.colors)

# Rasterize
cell = 0.01
xy_min = pcd_rot_xy.min(axis=0)
xy_max = pcd_rot_xy.max(axis=0)
nx = int(np.ceil((xy_max[0] - xy_min[0]) / cell))
ny = int(np.ceil((xy_max[1] - xy_min[1]) / cell))
# Remove boundaries (4% margin)
xy_min += 0.04 * (xy_max - xy_min)
xy_max -= 0.04 * (xy_max - xy_min)
nx = int(np.ceil((xy_max[0] - xy_min[0]) / cell))
ny = int(np.ceil((xy_max[1] - xy_min[1]) / cell))
## Map points to grid indices
ix = np.clip(((pcd_rot_xy[:,0] - xy_min[0]) / cell).astype(int), 0, nx-1)
iy = np.clip(((pcd_rot_xy[:,1] - xy_min[1]) / cell).astype(int), 0, ny-1)
lin = ix + iy * nx
## Height map
hmin = np.full(nx*ny, np.nan, dtype=float)
hmean = np.full(nx*ny, np.nan, dtype=float)
rgb = np.zeros((nx*ny, 3), dtype=float)
count = np.zeros(nx*ny, dtype=int)
for idx, color, z_val in zip(lin, pcd_rot_colors, pcd_rot_z):
    if np.isnan(hmin[idx]) or z_val < hmin[idx]:
        hmin[idx] = z_val
        rgb[idx] = color
    if np.isnan(hmean[idx]):
        hmean[idx] = z_val
    else:
        hmean[idx] = 0.5*(hmean[idx] + z_val)  # simple streaming approx
    count[idx] += 1
Hmin = hmin.reshape(ny, nx)
Hmean = hmean.reshape(ny, nx)
Count = count.reshape(ny, nx)

# Visualize rgb map
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.title("RGB Map")
plt.imshow(rgb.reshape(ny, nx, 3), origin='lower')
plt.show()
exit(-1)

H = Hmin.copy()
mask_valid = ~np.isnan(H)
if mask_valid.any():
    # quick inpainting by nearest valid (simple)
    from scipy.ndimage import distance_transform_edt
    # indices of nearest valid cell
    dist, (iy0, ix0) = distance_transform_edt(~mask_valid, return_indices=True)
    H[~mask_valid] = H[iy0[~mask_valid], ix0[~mask_valid]]

# Visualize H
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Height Map (min)")
# plt.imshow(Hmin, cmap='gray', origin='lower')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.title("Height Map (mean)")
# plt.imshow(H, cmap='gray', origin='lower')
# plt.colorbar()
# plt.show()
# exit(-1)

# Gradients (Sobel-like finite differences)
gx = np.zeros_like(H); gy = np.zeros_like(H)
gx[:,1:-1] = (H[:,2:] - H[:,:-2]) * 0.5 / cell
gy[1:-1,:] = (H[2:,:] - H[:-2,:]) * 0.5 / cell
grad_mag = np.hypot(gx, gy)

# Local density map (per m² if you want): points per cell / area
density = Count / (cell*cell)

# Negative deviation threshold (depth): cells lower than neighborhood median by >= depth_mm
depth_mm = 4.0  # try 3–8 mm depending on scan fidelity
depth = -depth_mm / 1000.0

# Compute global plane level ~0; use Hmean median as reference
ref = np.nanmedian(Hmean)
depression = (H - ref) < depth

# Strong gradient (crack edges)
grad_thr = np.percentile(grad_mag[mask_valid], 85) if mask_valid.any() else 0.02
edgey = grad_mag > grad_thr

# Low density (gaps / missing data)
dens_thr = np.percentile(density[mask_valid], 10) if mask_valid.any() else 1.0
sparse = density < max(dens_thr, 1.0)  # ensure at least 1 pt/cell

# Crack candidate: depression AND edge (thin valleys)
crack_candidate = depression

# Gap candidate: very low density in a contiguous region
gap_candidate = sparse & ~depression  # low density but not necessarily a valley

# Optional morphology to thin/smooth
crack_mask = morphology.binary_opening(crack_candidate, morphology.disk(1))
crack_skel = morphology.skeletonize(crack_mask)
gap_mask = morphology.binary_closing(gap_candidate, morphology.disk(2))

# Build a look-up from grid cell -> label
crack_cells = set(np.flatnonzero(crack_skel.ravel()))
gap_cells   = set(np.flatnonzero(gap_mask.ravel()))

# print("Number of points in point cloud: ", len(pcd_rot_points), ", number of pixels in rasterized image: ", H.size)
# print(type(crack_cells), len(crack_cells))

labels = np.zeros(pcd_rot_points.shape[0], dtype=int)  # 0=ok, 1=crack, 2=gap
for i, idx in enumerate(lin):
    if idx in crack_cells:
        labels[i] = 1
    # elif idx in gap_cells:
    #     labels[i] = 2

# Colorize
colors = np.zeros((pcd_rot_points.shape[0], 3))
colors[:] = pcd_rot.colors  # original color
colors[labels==1] = [1.0, 0.0, 0.0]      # red crack
colors[labels==2] = [0.0, 0.0, 1.0]      # blue gap

pcd_vis = o3d.geometry.PointCloud()
pcd_vis.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))  # original orientation
pcd_vis.colors = o3d.utility.Vector3dVector(colors)

# Translate original pcd for comparison
# pcd_translated = pcd_rot.translate((5, 0, 0))

plane_mesh_viz = get_viz_with_transparency(mesh=plane_mesh, name='Fitted Plane')
o3d.visualization.draw([pcd_vis, plane_mesh_viz])