import open3d as o3d
import open3d.visualization as vis
import numpy as np

def get_plane_mesh(pcd, plane_model, inliers, side_length = 3, color=[0, 0, 1]):
    inlier_points = np.array(pcd.select_by_index(inliers).points)
    centroid = np.mean(inlier_points, axis=0)

    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c], dtype=float)
    
    # Get the plane tangent
    plane_tangent = np.array([1, 0, 0], dtype=float)
    if plane_normal[0] > 0.9:
        plane_tangent = np.array([0, 1, 0], dtype=float)
    plane_tangent -= plane_normal.dot(plane_tangent) * plane_normal
    plane_tangent /= np.linalg.norm(plane_tangent)
    plane_cross_tangent = np.cross(plane_normal, plane_tangent)

    half = side_length/2
    corners = [
        centroid + half * ( plane_tangent + plane_cross_tangent),
        centroid + half * (-plane_tangent + plane_cross_tangent),
        centroid + half * (-plane_tangent - plane_cross_tangent),
        centroid + half * ( plane_tangent - plane_cross_tangent),
    ]
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])
    mesh.paint_uniform_color(color)
    mesh.compute_triangle_normals()

    return mesh

def get_viz_with_transparency(mesh, name = 'Plane'):
    # https://github.com/isl-org/Open3D/issues/2890
    mat_box = vis.rendering.MaterialRecord()
    mat_box.shader = 'defaultLitTransparency'
    mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
    mat_box.base_roughness = 0.0
    mat_box.base_reflectance = 0.0
    # mat_box.base_clearcoat = 1.0
    mat_box.thickness = 1.0
    mat_box.transmission = 1.0
    mat_box.absorption_distance = 10
    mat_box.absorption_color = [0.5, 0.5, 0.5]
    
    geom = {'name': name, 'geometry': mesh, 'material': mat_box}
    return geom