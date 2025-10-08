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
                 num_points: int, num_issues: int):
        self.distance_arr = distance_arr
        self.angle_arr = angle_arr
        self.angle_median = angle_median
        self.angle_mad = angle_mad
        self.angle_threshold = angle_threshold
        self.num_points = num_points
        self.num_issues = num_issues

def check_integrity(pcd_file_path: str, *, 
                 depth_thr: float = 0.004, near_plane_thr: float = 0.05, 
                 min_angle_thr: float = 15.0,
                 dbscan_eps: float = 0.05, dbscan_min_samples: int = 5,
                 issue_percent_thr: float = 0.005) -> tuple[o3d.geometry.PointCloud, bool, IntegrityDetails]:
    """
    Analyzes the given point cloud file for surface integrity issues.
    """
    pcd_original = o3d.io.read_point_cloud(pcd_file_path)
    pcd = pcd_original.voxel_down_sample(voxel_size=0.01)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(50) # Hardcoded for now
    
    # Recompute normals
    num_points = np.asarray(pcd.points).shape[0]
    print(f"Number of points in the point cloud: {num_points}")
    
    # Fit a plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    # plane_mesh = get_plane_mesh(pcd, plane_model, inliers, side_length=5)
    # ground = pcd.select_by_index(inliers)
    # non_ground = pcd.select_by_index(inliers, invert=True)
    
    # Flatten to 2D by aligning to Z-axis
    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c], dtype=float)
    
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
    
    # Establish thresholds
    # Angle: robust data-driven (median + 3*MAD), with a floor like 15°
    near_plane = np.abs(signed_dist) < near_plane_thr  # optional gate to ignore tall objects; tune or drop
    base = angle_deg[near_plane] if np.any(near_plane) else angle_deg
    med = np.median(base)
    mad = 1.4826 * np.median(np.abs(base - med)) if base.size else 0.0
    angle_thr = max(min_angle_thr, med + 3*mad)
    
    angle_issue_mask = angle_deg > angle_thr
    depth_issue_mask = True#(signed_dist < -depth_thr)
    issue_mask = angle_issue_mask & depth_issue_mask
    
    # Cluster issues
    xy = P[:, :2]
    idx = np.flatnonzero(issue_mask)
    if idx.size:
        cl = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(xy[idx])
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

    # plane_mesh_geom = get_viz_with_transparency(mesh=plane_mesh, name='Fitted Plane')
    
    total_points = len(pcd.points)
    issue_points = np.sum(issue_mask)
    issue_percent = issue_points / total_points if total_points > 0 else 0
    print(f"Issue points: {issue_points}/{total_points} ({issue_percent*100:.2f}%)")
    
    pcd_integrity_details = IntegrityDetails(
        distance_arr=signed_dist,
        angle_arr=angle_deg,
        angle_median=med,
        angle_mad=mad,
        angle_threshold=angle_thr,
        num_points=num_points,
        num_issues=issue_points
    )

    return pcd_vis, issue_percent > issue_percent_thr, pcd_integrity_details