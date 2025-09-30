"""
This script subsamples a point cloud by z-axis. 
"""
import os
import numpy as np
import open3d as o3d
from skimage import morphology, measure
import random

# Set random seed for reproducibility
random.seed(42)

from utils.plane import get_plane_mesh, get_viz_with_transparency

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "lab_controlled", "adhoc"))
PCD_FILE_PATH = os.path.join(DATASET_PATH, "4_1_0_1_bottom_cropped.ply")
pcd = o3d.io.read_point_cloud(PCD_FILE_PATH)

print(len(pcd.points), "points in the original point cloud")

# o3d.visualization.draw_geometries_with_editing([pcd])
# exit(-1)

# Downsample
# pcd = pcd_original.voxel_down_sample(voxel_size=0.001)

# o3d.visualization.draw_geometries_with_editing([pcd_original])
# exit(-1)

# Remove statistical outliers (tune nb_neighbors/std_ratio for your density)
# pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

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
# o3d.visualization.draw_geometries([pcd_rot, plane_rot_mesh])

pcd_rot_points = np.asarray(pcd_rot.points)
pcd_rot_xy = pcd_rot_points[:, :2]
pcd_rot_z = pcd_rot_points[:, 2]

# Check how many unique x,y values exist if rounded to 1mm
# xy_rounded = np.round(pcd_rot_xy * 1000).astype(int)
# print("Unique xy (1mm):", np.unique(xy_rounded, axis=0).shape[0], "out of", pcd_rot_xy.shape[0])
# exit(-1)

pcd_rot_colors = np.asarray(pcd_rot.colors)

# Shuffle
pcd_rot_data = np.stack((pcd_rot_points, pcd_rot_colors), axis=1)
pcd_rot_data_shuffled = pcd_rot_data[np.random.permutation(pcd_rot_data.shape[0])]
pcd_rot_points_randomized = pcd_rot_data_shuffled[:, 0, :]
pcd_rot_colors_randomized = pcd_rot_data_shuffled[:, 1, :]

cell = 0.001 # 1 mm
xy_min = pcd_rot_xy.min(axis=0)
xy_max = pcd_rot_xy.max(axis=0)
nx = int(np.ceil((xy_max[0] - xy_min[0]) / cell))
ny = int(np.ceil((xy_max[1] - xy_min[1]) / cell))
print("Grid size:", nx, ny)
## Map points to grid indices
ix = np.clip(((pcd_rot_xy[:,0] - xy_min[0]) / cell).astype(int), 0, nx-1)
iy = np.clip(((pcd_rot_xy[:,1] - xy_min[1]) / cell).astype(int), 0, ny-1)
lin = ix + iy * nx
## Buffer for min/max z; create new point cloud with subsampled points
pcd_rot_points_buffer = []
pcd_rot_colors_buffer = []
# z = np.full((nx*ny, 1), np.nan, dtype=float)  # store up to 1 value per cell
# for idx, z_val, z_color in zip(lin, pcd_rot_points_randomized, pcd_rot_colors_randomized):
#     col = np.where(np.isnan(z[idx]))[0]
#     if col.size == 0: continue
#     z[idx, col[0]] = z_val[2]
#     pcd_rot_points_buffer.append(z_val)
#     pcd_rot_colors_buffer.append(z_color)
z_min = np.full((nx*ny, 1), np.nan, dtype=float)  # store min z per cell
for idx, z_val, z_color in zip(lin, pcd_rot_points_randomized, pcd_rot_colors_randomized):
    if np.isnan(z_min[idx]):
        z_min[idx] = z_val[2]
        pcd_rot_points_buffer.append(z_val)
        pcd_rot_colors_buffer.append(z_color)
    elif z_val[2] < z_min[idx]:
        z_min[idx] = z_val[2]
        pcd_rot_points_buffer[-1] = z_val
        pcd_rot_colors_buffer[-1] = z_color

pcd_rot_points_buffer = np.array(pcd_rot_points_buffer)
pcd_rot_colors_buffer = np.array(pcd_rot_colors_buffer)
print(f"Subsampled from {len(pcd_rot_points_randomized)} to {len(pcd_rot_points_buffer)} points")
pcd_rot_subsampled = o3d.geometry.PointCloud()
pcd_rot_subsampled.points = o3d.utility.Vector3dVector(pcd_rot_points_buffer)
pcd_rot_subsampled.colors = o3d.utility.Vector3dVector(pcd_rot_colors_buffer)

# Visualize
# o3d.visualization.draw_geometries([pcd_rot, pcd_rot_subsampled, plane_rot_mesh])
o3d.visualization.draw_geometries_with_editing([pcd_rot_subsampled])
exit(-1)
