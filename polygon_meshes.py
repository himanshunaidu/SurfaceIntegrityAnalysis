import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load the mesh and sample to point cloud ---
pcd = o3d.io.read_point_cloud("dataset/ios_point_cloud/Sidewalk1.ply")
# pcd = o3d.io.read_point_cloud("dataset/deepwalk/0_138311.0.ply")
# o3d.visualization.draw_geometries([pcd])
# uniformly sample points from the point cloud
pcd = pcd.uniform_down_sample(every_k_points=5)

# Step 2: Fit plane
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model

# Step 3: Measure deviation
points = np.asarray(pcd.points)
distances = np.abs((a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) /
                   np.sqrt(a ** 2 + b ** 2 + c ** 2))

# Step 4: Color crack candidates
threshold = 0.02
crack_mask = distances > threshold
# colors = np.tile([0.5, 0.5, 0.5], (len(points), 1))

if pcd.has_colors():
    colors = np.asarray(pcd.colors).copy()
else:
    # Default gray if no color present
    colors = np.tile([0.5, 0.5, 0.5], (len(points), 1))

colors[crack_mask] = colors[crack_mask]+[0.5, 0.0, 0.0]
pcd.colors = o3d.utility.Vector3dVector(colors)


# --- Step 5: Visualize the plane ---
# Get centroid and PCA to define plane extent
inlier_points = points[inliers]
centroid = inlier_points.mean(axis=0)

# PCA for local plane orientation
cov = np.cov(inlier_points.T)
eigvals, eigvecs = np.linalg.eigh(cov)
plane_axes = eigvecs[:, [1, 2]]  # Use 2 main axes in the plane

# Define rectangle size (manually or based on data spread)
width = 2.0  # meters
height = 2.0

# Generate corners of the rectangle in the plane
corners = np.array([
    [-width / 2, -height / 2],
    [ width / 2, -height / 2],
    [ width / 2,  height / 2],
    [-width / 2,  height / 2],
])

# Map to 3D points on the plane
plane_points = centroid + corners @ plane_axes.T

# Create mesh from plane corners
plane_mesh = o3d.geometry.TriangleMesh()
plane_mesh.vertices = o3d.utility.Vector3dVector(plane_points)
plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]])
plane_mesh.compute_vertex_normals()
plane_mesh.paint_uniform_color([0.0, 1.0, 0.0])  # green
plane_mesh.translate([0, 0, 0])
plane_mesh.compute_triangle_normals()


# Step 5: Visualize
o3d.visualization.draw_geometries([pcd, plane_mesh])