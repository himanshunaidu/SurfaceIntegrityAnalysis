import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "ios_point_cloud"))
PATH = os.path.join(DATASET_PATH, "Sidewalk1.ply")

# --- Step 1: Load the mesh and sample to point cloud ---
pcd = o3d.io.read_point_cloud(PATH)
# o3d.visualization.draw_geometries([pcd])
# uniformly sample points from the point cloud
pcd = pcd.uniform_down_sample(every_k_points=10)
print("Loaded point cloud and downsampled.")

# --- Step 2: Estimate normals (optional, not used in meshing directly) ---
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(50)
print("Estimated normals for point cloud.")

# --- Step 3: Convert to triangle mesh using Ball Pivoting ---
radii = [0.01, 0.015, 0.02]  # Adjust based on your point spacing
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)
print("Created triangle mesh from point cloud.")

# --- Step 4: Compute face normals ---
mesh.compute_triangle_normals()
triangle_normals = np.asarray(mesh.triangle_normals)
print("Computed triangle normals for mesh.")

# --- Step 5: Calculate the mean normal vector ---
mean_normal = triangle_normals.mean(axis=0)
mean_normal /= np.linalg.norm(mean_normal)  # Normalize
print("Mean normal vector:", mean_normal)

# --- Step 6: Compute angular deviation from mean normal ---
dot_products = triangle_normals @ mean_normal
dot_products = np.clip(dot_products, -1.0, 1.0)  # Clamp to valid arccos range
angles = np.arccos(dot_products)  # Radians

# Convert to degrees for easier thresholding
angles_deg = np.degrees(angles)
print("Computed angles in degrees from mean normal.")

# --- Step 7: Flag anomalies based on angle deviation threshold ---
threshold_deg = 15  # You can adjust this
crack_mask = angles_deg > threshold_deg
print(f"Flagged {np.sum(crack_mask)} triangles as anomalies based on angle threshold of {threshold_deg} degrees.")

# --- Step 8: Color the triangles ---
def mesh_with_per_face_color(original_mesh, face_colors):
    triangles = np.asarray(original_mesh.triangles)
    vertices = np.asarray(original_mesh.vertices)

    new_vertices = []
    new_triangles = []
    new_colors = []

    for i, tri in enumerate(triangles):
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        base_index = len(new_vertices)
        new_vertices.extend([v0, v1, v2])
        new_triangles.append([base_index, base_index + 1, base_index + 2])
        new_colors.extend([face_colors[i]] * 3)

    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
    mesh_out.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
    mesh_out.vertex_colors = o3d.utility.Vector3dVector(np.array(new_colors))
    return mesh_out

face_colors = np.tile([0.6, 0.6, 0.6], (len(mesh.triangles), 1))  # default gray
face_colors[crack_mask] = [1.0, 0.0, 0.0]  # mark suspect faces red
mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(mesh.triangles))
mesh.triangle_uvs = o3d.utility.Vector2dVector()
mesh.textures = []
# mesh.triangle_colors = o3d.utility.Vector3dVector(face_colors)
colored_mesh = mesh_with_per_face_color(mesh, face_colors)
print("Colored triangles based on anomaly detection.")

# --- Step 9: Visualize ---
o3d.visualization.draw_geometries([colored_mesh])