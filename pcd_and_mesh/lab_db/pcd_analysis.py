"""
This script analyzes a 3D point cloud to check for surface integrity issues.
"""
import os
import glob
import open3d as o3d
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import logging

from utils.plane import get_plane_mesh, get_viz_with_transparency
from utils.stats import get_array_stats
from schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
from schema_utils import load_lab_db_frames

def update_lab_db_frames(dataset_path: str, db_main_frame: pd.DataFrame, db_results_frame: pd.DataFrame) -> None:
    """
    Updates the lab database CSV files from the given pandas DataFrames.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Database file not found: {dataset_path}")

    db_main_path = os.path.join(dataset_path, "lab_db.csv")
    db_main_frame.to_csv(db_main_path, index=False)
    print(f"Updated main database CSV file: {db_main_path}")

    db_results_path = os.path.join(dataset_path, "lab_db_results.csv")
    db_results_frame.to_csv(db_results_path, index=False)
    print(f"Updated results database CSV file: {db_results_path}")

def process_pcd_files(dataset_path: str, db_results_frame: pd.DataFrame, *, 
                      pcd_dir: str = 'pcd_cropped', output_dir: str = 'pcd_analysis') -> None:
    """
    Loads all point cloud files from the dataset path based on the database results frame and analyzes them for
    surface integrity issues.

    Note: The db_results_frame contains column 'trial_name' based on which the point cloud files are named (with some differences).
    Each row in db_results_frame corresponds to multiple point clouds in dataset_path.
        (Multiple point clouds because of repeated trials with same parameters. We can call them subtrials.)

    The subtrials are named as per the 'trial_name' column in db_results_frame, along with 'a', 'b', ... suffixes for repeated trials.
    """
    point_cloud_files = glob.glob(os.path.join(dataset_path, pcd_dir, "*.ply"))
    point_cloud_files.sort()
    point_cloud_file_names = [os.path.basename(f) for f in point_cloud_files]
    print(f"Found {len(point_cloud_file_names)} point cloud files in {os.path.join(dataset_path, pcd_dir)}")
    
    output_dir_path = os.path.join(dataset_path, output_dir)
    os.makedirs(output_dir_path, exist_ok=True)
    
    for index, row in db_results_frame.iterrows():
        trial_name = row['trial_name']
        num_issues = 0
        row_pcds = [f for f in point_cloud_file_names if f.startswith(trial_name)]
        if not row_pcds:
            # print(f"No point cloud file found for trial_name '{trial_name}' in row index {index}")
            continue
        print(f"Found {len(row_pcds)} point cloud files for trial_name '{trial_name}' in row index {index}")
        
        for pcd_file in row_pcds:
            pcd_file_path = os.path.join(dataset_path, pcd_dir, pcd_file)
            logging.info(f"Processing point cloud file: {pcd_file_path}")
            
            analyzed_pcd, has_issues = check_pcd_integrity(pcd_file_path)
            
            output_pcd_file_path = os.path.join(output_dir_path, pcd_file)
            o3d.io.write_point_cloud(output_pcd_file_path, analyzed_pcd)
            print(f"Saved analyzed point cloud to: {output_pcd_file_path}")
            
            if has_issues: num_issues += 1
        db_results_frame.at[index, ResultColumns.POINT_CLOUD_RESULT.value] = num_issues
        print(f"Updated DataFrame for row index {index}, trial_name '{trial_name}': point_cloud_result = {num_issues}")

def check_pcd_integrity(pcd_file_path: str, *, 
                 depth_thr: float = 0.004, near_plane_thr: float = 0.05, 
                 min_angle_thr: float = 15.0,
                 dbscan_eps: float = 0.05, dbscan_min_samples: int = 5,
                 issue_percent_thr: float = 0.005) -> tuple[o3d.geometry.PointCloud, bool]:
    """
    Analyzes the given point cloud file for surface integrity issues.
    """
    pcd_original = o3d.io.read_point_cloud(pcd_file_path)
    pcd = pcd_original.voxel_down_sample(voxel_size=0.01)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(50) # Hardcoded for now
    
    # Recompute normals
    num_points = np.asarray(pcd.points).shape[0]
    logging.info(f"Number of points in the point cloud: {num_points}")
    
    # Fit a plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    plane_mesh = get_plane_mesh(pcd, plane_model, inliers, side_length=5)
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
    logging.info(f"Angle stats: {get_array_stats(angle_deg)}")
    
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
    logging.info(f"Issue points: {issue_points}/{total_points} ({issue_percent*100:.2f}%)")

    return pcd_vis, issue_percent > issue_percent_thr
    
if __name__=="__main__":
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled", "experiment_4"))
    
    logging.basicConfig(
        filename=os.path.join(DATASET_PATH, "pcd_analysis.log"),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    db_main_frame, db_results_frame = load_lab_db_frames(DATASET_PATH)
    print(f"Loaded lab database with {len(db_main_frame)} entries.")
    
    process_pcd_files(DATASET_PATH, db_results_frame)
    
    update_lab_db_frames(DATASET_PATH, db_main_frame, db_results_frame)