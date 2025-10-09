import os
import glob
import open3d as o3d
import numpy as np
import pandas as pd
import logging
from sklearn.cluster import DBSCAN

class IntegrityDetails:
    def __init__(self, distance_arr: np.ndarray, angle_arr: np.ndarray, 
                 angle_median: float, angle_mad: float, angle_threshold: float,
                 num_polygons: int, num_issues: int, 
                 total_area: float, issue_area: float):
        self.distance_arr = distance_arr
        self.angle_arr = angle_arr
        self.angle_median = angle_median
        self.angle_mad = angle_mad
        self.angle_threshold = angle_threshold
        self.num_polygons = num_polygons
        self.num_issues = num_issues
        self.total_area = total_area
        self.issue_area = issue_area


def check_integrity(mesh_file_path: str, *, 
                 depth_thr: float = 0.005, near_plane_thr: float = 0.05, 
                 min_angle_thr: float = 10.0,
                 dbscan_eps: float = 0.05, dbscan_min_samples: int = 3,
                 issue_area_percent_thr: float = 0.005) -> tuple[o3d.geometry.TriangleMesh, bool, IntegrityDetails]:
    """
    Analyzes a 3D mesh file for surface integrity issues.
    """    
    # --- Step 1: Load the mesh ---
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    # --- Step 2: Analyze the mesh ---
    num_polygons = len(mesh.triangles)
    pcd = mesh.sample_points_uniformly(number_of_points=num_polygons * 10)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(50)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
    # plane_mesh = get_plane_mesh(pcd, plane_model, inliers, side_length=5)
    
    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c], dtype=float)

    triangle_centroids = np.mean(np.asarray(mesh.vertices)[np.asarray(mesh.triangles)], axis=1)
    # signed distance of triangle centroids to plane
    signed_dist = triangle_centroids @ plane_normal + d  # meters; negative = below plane

    N = np.asarray(mesh.triangle_normals)
    flip = (N @ plane_normal) < 0
    N[flip] *= -1
    mesh.triangle_normals = o3d.utility.Vector3dVector(N)

    cosang = np.clip(N @ plane_normal, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosang))
    
    near_plane = np.abs(signed_dist) < near_plane_thr  # optional gate to ignore tall objects; tune or drop
    base = angle_deg[near_plane] if np.any(near_plane) else angle_deg
    med = np.median(base)
    mad = 1.4826 * np.median(np.abs(base - med)) if base.size else 0.0
    angle_thr = max(min_angle_thr, med + 3*mad)
    
    unsigned_dist = np.abs(signed_dist)
    angle_issue_mask = angle_deg > angle_thr
    depth_issue_mask = unsigned_dist > depth_thr
    issue_mask = angle_issue_mask & depth_issue_mask
    
    T = np.asarray(mesh.triangles)
    T_xy = triangle_centroids[:, :2]  # already in original frame; flattening not required
    idx = np.flatnonzero(issue_mask)
    if idx.size:
        cl = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(T_xy[idx])
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

    total_area = 0.0
    issue_area = 0.0
    for i in range(len(T)):
        triangle = T[i]
        v0 = np.asarray(mesh.vertices)[triangle[0]]
        v1 = np.asarray(mesh.vertices)[triangle[1]]
        v2 = np.asarray(mesh.vertices)[triangle[2]]
        tri_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
        if issue_mask[i]:
            vertex_colors[T[i]] = [1.0, 0.0, 0.0]  # red
            issue_area += tri_area
        total_area += tri_area
        
    issue_area_percent = issue_area / total_area if total_area > 0 else 0.0
    print(f"Issue area: {issue_area:.4f} m^2 ({issue_area_percent*100:.2f}%)")
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    # o3d.visualization.draw_geometries([mesh])
    
    mesh_integrity_details = IntegrityDetails(
        distance_arr=signed_dist,
        angle_arr=angle_deg,
        angle_median=med,
        angle_mad=mad,
        angle_threshold=angle_thr,
        num_polygons=num_polygons,
        num_issues=np.sum(issue_mask),
        total_area=total_area,
        issue_area=issue_area
    )

    return mesh, (issue_area_percent > issue_area_percent_thr), mesh_integrity_details