"""
This script analyzes a 3D mesh to check for surface integrity issues.
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

def process_mesh_files(dataset_path: str, db_results_frame: pd.DataFrame, *, 
                    mesh_dir: str = 'mesh_cropped', output_dir: str = 'mesh_analyzed') -> None:
    """
    Loads all mesh files from the dataset path based on the database results frame and analyzes them for
    surface integrity issues.
    
    Note: The db_results_frame contains column 'trial_name' based on which the mesh files are named (with some differences).
    Each row in db_results_frame corresponds to multiple meshes in dataset_path.
        (Multiple meshes because of repeated trials with same parameters. We can call them subtrials.)
    The subtrials are named as per the 'trial_name' column in db_results_frame, along with 'a', 'b', ... suffixes for repeated trials.
    """
    subtrial_directories = glob.glob(os.path.join(dataset_path, "*"))
    subtrial_directories.sort()
    subtrial_directory_names = [os.path.basename(d) for d in subtrial_directories if os.path.isdir(d)]
    print(f"Found {len(subtrial_directory_names)} subtrial directories in {dataset_path}")
    
    for index, row in db_results_frame.iterrows():
        trial_name = row['trial_name']
        num_issues = 0
        row_subtrials = [d for d in subtrial_directory_names if d.startswith(trial_name)]
        if not row_subtrials:
            # print(f"No subdirectory found for trial_name '{trial_name}' in row index {index}")
            continue
        for sub_dir_name in row_subtrials:
            # Setup mesh analysis for this subdirectory
            sub_dir_path = os.path.join(dataset_path, sub_dir_name)
            mesh_path = os.path.join(sub_dir_path, mesh_dir)
            mesh_files = glob.glob(os.path.join(mesh_path, "*.ply"))
            if not mesh_files:
                print(f"No mesh files found in {mesh_path} for trial_name '{trial_name}' in row index {index}")
                continue
            mesh_file = mesh_files[0]  # Assuming one .ply file per subtrial
            logging.info(f"Analyzing mesh file {mesh_file} for trial_name '{trial_name}' in row index {index}")
            
            analyzed_mesh, has_issues = check_mesh_integrity(mesh_file)
            
            output_subdir = os.path.join(sub_dir_path, output_dir)
            os.makedirs(output_subdir, exist_ok=True)
            output_mesh_file = os.path.join(output_subdir, f"{os.path.basename(mesh_file)}")
            o3d.io.write_triangle_mesh(output_mesh_file, analyzed_mesh)
            print(f"Wrote analyzed mesh to {output_mesh_file}, Issues found: {has_issues}")
            
            if has_issues: num_issues += 1
            # current_polygon_mesh_result = db_results_frame.at[index, ResultColumns.POLYGON_MESH_RESULT.value]
            # if pd.isna(current_polygon_mesh_result) or current_polygon_mesh_result == "":
            #     current_polygon_mesh_result = 0
            # db_results_frame.at[index, ResultColumns.POLYGON_MESH_RESULT.value] = current_polygon_mesh_result + (1 if has_issues else 0)
        db_results_frame.at[index, ResultColumns.POLYGON_MESH_RESULT.value] = num_issues
        print(f"Completed processing for row index {index}, trial_name '{trial_name}'. {num_issues} subtrials with issues.")

def check_mesh_integrity(mesh_file_path: str, *, 
                 depth_thr: float = 0.005, near_plane_thr: float = 0.05, 
                 min_angle_thr: float = 10.0,
                 dbscan_eps: float = 0.05, dbscan_min_samples: int = 3,
                 issue_area_percent_thr: float = 0.005) -> tuple[o3d.geometry.TriangleMesh, bool]:
    """
    Analyzes a 3D mesh file for surface integrity issues.
    """
    # --- Step 1: Load the mesh ---
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    logging.info(f"Loaded mesh from: {mesh_file_path}")
    
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
    logging.info(f"Signed distance stats: {get_array_stats(signed_dist)}")

    N = np.asarray(mesh.triangle_normals)
    flip = (N @ plane_normal) < 0
    N[flip] *= -1
    mesh.triangle_normals = o3d.utility.Vector3dVector(N)

    cosang = np.clip(N @ plane_normal, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosang))
    logging.info(f"Angle stats: {get_array_stats(angle_deg)}")
    
    near_plane = np.abs(signed_dist) < near_plane_thr  # optional gate to ignore tall objects; tune or drop
    base = angle_deg[near_plane] if np.any(near_plane) else angle_deg
    med = np.median(base)
    mad = 1.4826 * np.median(np.abs(base - med)) if base.size else 0.0
    logging.info(f"Angle median = {med:.2f} deg, MAD = {mad:.2f} deg")
    angle_thr = max(min_angle_thr, med + 3*mad)

    logging.info(f"Angle threshold = {angle_thr} deg")
    logging.info(f"Depth threshold = {depth_thr} m")
    
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
        
    logging.info(f"Found {np.sum(issue_mask)} triangles with issues ({np.sum(issue_mask)/num_polygons*100:.1f}%)")
    
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
    logging.info(f"Issue area: {issue_area:.4f} m^2 ({issue_area_percent*100:.2f}%)")
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    # o3d.visualization.draw_geometries([mesh])

    return mesh, (issue_area_percent > issue_area_percent_thr)

if __name__=="__main__":
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled", "experiment_4"))
    
    logging.basicConfig(
        filename=os.path.join(DATASET_PATH, "mesh_analysis.log"),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    db_main_frame, db_results_frame = load_lab_db_frames(DATASET_PATH)
    print(f"Loaded lab database with {len(db_main_frame)} entries.")
    
    dataset_mesh_path = os.path.join(DATASET_PATH, "main")
    process_mesh_files(dataset_mesh_path, db_results_frame)

    update_lab_db_frames(DATASET_PATH, db_main_frame, db_results_frame)