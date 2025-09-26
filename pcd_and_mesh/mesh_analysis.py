import os
import glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from utils.plane import get_plane_mesh, get_viz_with_transparency
from utils.stats import get_array_stats

# DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "lab_controlled", "experiment_2"))
# ROW_PATH = os.path.join(DATASET_PATH, "main", "0_1_0_1")
# MESH_PATH = os.path.join(ROW_PATH, "mesh_cropped")
# MESH_FILE_PATH = glob.glob(os.path.join(MESH_PATH, "*.ply"))[0]  # Assuming one .ply file per row
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "lab_controlled", "adhoc"))
# MESH_FILE_PATH = os.path.join(DATASET_PATH, "Experiment_1/4_1_0_1_adhoc_mesh/mesh/EC132D92-22C4-446D-AA2D-8B174BAE87D4.ply")
MESH_FILE_PATH = os.path.join(DATASET_PATH, "Experiment_1/4_1_0_1_adhoc_mesh/mesh_cropped/cropped_1.ply")

# --- Step 1: Load the mesh ---
mesh = o3d.io.read_triangle_mesh(MESH_FILE_PATH)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()
print("Loaded mesh from:", MESH_FILE_PATH)

# Visualize the mesh
# o3d.visualization.draw_geometries_with_editing([mesh])
# exit(-1)

# --- Step 2: Analyze the mesh ---
num_polygons = len(mesh.triangles)
pcd = mesh.sample_points_uniformly(number_of_points=num_polygons * 10)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(50)
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
plane_mesh = get_plane_mesh(pcd, plane_model, inliers, side_length=5)

[a, b, c, d] = plane_model
plane_normal = np.array([a, b, c], dtype=float)

triangle_centroids = np.mean(np.asarray(mesh.vertices)[np.asarray(mesh.triangles)], axis=1)
# signed distance of triangle centroids to plane
signed_dist = triangle_centroids @ plane_normal + d  # meters; negative = below plane
print(get_array_stats(signed_dist))

N = np.asarray(mesh.triangle_normals)
flip = (N @ plane_normal) < 0
N[flip] *= -1
mesh.triangle_normals = o3d.utility.Vector3dVector(N)

cosang = np.clip(N @ plane_normal, -1.0, 1.0)
angle_deg = np.degrees(np.arccos(cosang))
print(get_array_stats(angle_deg))

# Get thresholds using robust statistics (median + MAD)
depth_thr = 0.004   # meters
near_plane = np.abs(signed_dist) < 0.05  # optional gate to ignore tall objects; tune or drop
base = angle_deg[near_plane] if np.any(near_plane) else angle_deg
med = np.median(base)
mad = 1.4826 * np.median(np.abs(base - med)) if base.size else 0.0
angle_thr = max(5, med + 3*mad)

print(f"Depth threshold = {depth_thr} m")
print(f"Angle threshold = {angle_thr} deg")

unsigned_dist = np.abs(signed_dist)
angle_issue_mask = angle_deg > angle_thr
depth_issue_mask = True#unsigned_dist > depth_thr
issue_mask = angle_issue_mask & depth_issue_mask

T = np.asarray(mesh.triangles)
T_xy = triangle_centroids[:, :2]  # already in original frame; flattening not required
idx = np.flatnonzero(issue_mask)
if idx.size:
    cl = DBSCAN(eps=0.02, min_samples=5).fit(T_xy[idx])
    keep = cl.labels_ >= 0
    hard_mask = np.zeros_like(issue_mask)
    hard_mask[idx[keep]] = True
    issue_mask = hard_mask
    
print(f"Found {np.sum(issue_mask)} triangles with issues ({np.sum(issue_mask)/num_polygons*100:.1f}%)")

# Color all vertices based on triangle issues
vertex_colors = np.asarray(mesh.vertex_colors)
if vertex_colors.shape[0] != np.asarray(mesh.vertices).shape[0]:
    vertex_colors = np.ones_like(np.asarray(mesh.vertices)) * 0.5  # gray background
T = np.asarray(mesh.triangles)
for i in range(len(T)):
    if issue_mask[i]:
        vertex_colors[T[i]] = [1.0, 0.0, 0.0]  # red

mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
# o3d.visualization.draw([mesh, plane_mesh])
o3d.visualization.draw_geometries_with_editing([mesh])