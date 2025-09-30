import os
import glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from utils.plane import get_plane_mesh, get_viz_with_transparency
from utils.stats import get_array_stats

# DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "lab_controlled", "experiment_2"))
# ROW_PATH = os.path.join(DATASET_PATH, "pcd", "pcd_cropped")
# PCD_FILE_PATH = os.path.join(ROW_PATH, "4_1_0_1.ply")
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "lab_controlled", "adhoc"))
PCD_FILE_PATH = os.path.join(DATASET_PATH, "4_1_0_1_adhoc_cropped.ply")

# --- Step 1: Load the mesh ---
pcd_original = o3d.io.read_point_cloud(PCD_FILE_PATH)
print("Loaded pcd from:", PCD_FILE_PATH)

# Visualize the mesh
# o3d.visualization.draw_geometries_with_editing([pcd_original])
# exit(-1)

# --- Step 2: Analyze the mesh ---
# Downsample
pcd = pcd_original.voxel_down_sample(voxel_size=0.01)
# Remove statistical outliers (tune nb_neighbors/std_ratio for your density)
# pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

# Recompute normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(50)

num_points = np.asarray(pcd.points).shape[0]
print(f"Number of points in the point cloud: {num_points}")

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
# print("Plane model: ", plane_model)
# exit(-1)

# Per-point signed distance
P = np.asarray(pcd.points)
signed_dist = P @ plane_normal + d  # meters; negative = below plane

# Per-point normals
normal_radius = 0.03
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50))
pcd.orient_normals_consistent_tangent_plane(30)

N = np.asarray(pcd.normals)
# Make normals coherent w.r.t. the plane normal so angles are in [0, 90]
flip = (N @ plane_normal) < 0
N[flip] *= -1
pcd.normals = o3d.utility.Vector3dVector(N)

# Angle between point normal and plane normal (degrees)
cosang = np.clip(N @ plane_normal, -1.0, 1.0)
angle_deg = np.degrees(np.arccos(cosang))
print(get_array_stats(angle_deg))

# Establish thresholds
depth_thr = 0.004   # meters
# Angle: robust data-driven (median + 3*MAD), with a floor like 15°
near_plane = np.abs(signed_dist) < 0.05  # optional gate to ignore tall objects; tune or drop
base = angle_deg[near_plane] if np.any(near_plane) else angle_deg
med = np.median(base)
mad = 1.4826 * np.median(np.abs(base - med)) if base.size else 0.0
angle_thr = max(15.0, med + 3*mad)

# issue_mask = (angle_deg > angle_thr) | (signed_dist < -depth_thr)
angle_issue_mask = angle_deg > angle_thr
depth_issue_mask = True#(signed_dist < -depth_thr)
issue_mask = angle_issue_mask & depth_issue_mask

# Cluster issues
xy = P[:, :2]
idx = np.flatnonzero(issue_mask)
if idx.size:
    cl = DBSCAN(eps=0.02, min_samples=5).fit(xy[idx])
    keep = cl.labels_ >= 0
    hard_mask = np.zeros_like(issue_mask)
    hard_mask[idx[keep]] = True
    issue_mask = hard_mask

pcd_vis = o3d.geometry.PointCloud()
pcd_vis.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))

# --- Colorize ---
colors = np.asarray(pcd.colors)
# colors[issue_mask] = [0.0, 0.0, 1.0]  # blue = issues by dist or normals
color_1 = (angle_deg[issue_mask] / 90.0).reshape(-1, 1) * np.array([1.0, 0.0, 0.0]).reshape(1, -1)
color_2 = (1 - angle_deg[issue_mask] / 90.0).reshape(-1, 1) * np.array([0.0, 0.0, 1.0]).reshape(1, -1)
colors[issue_mask] = color_1 + color_2 # Note: gradient from red to blue
pcd_vis.colors = o3d.utility.Vector3dVector(colors)

plane_mesh_viz = get_viz_with_transparency(mesh=plane_mesh, name='Fitted Plane')
o3d.visualization.draw([pcd_vis, plane_mesh_viz])