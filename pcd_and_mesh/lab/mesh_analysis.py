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

from vision_utils.mesh import check_integrity as check_mesh_integrity, IntegrityDetails as MeshIntegrityDetails
from utils.plane import get_plane_mesh, get_viz_with_transparency
from utils.stats import get_array_stats
from db.schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
from db.read import load_data_frames
from db.update import update_data_frames

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
            
            analyzed_mesh, has_issues, mesh_integrity_details = check_mesh_integrity(mesh_file)
            logging.info(f"Signed distance stats: {get_array_stats(mesh_integrity_details.distance_arr)}")
            logging.info(f"Angle stats: {get_array_stats(mesh_integrity_details.angle_arr)}")
            logging.info(f"Angle median = {mesh_integrity_details.angle_median:.2f} deg, MAD = {mesh_integrity_details.angle_mad:.2f} deg")
            logging.info(f"Angle threshold = {mesh_integrity_details.angle_threshold} deg")
            logging.info(f"Depth threshold = {0.005} m") # TODO: Remove hardcoding
            logging.info(f"Found {mesh_integrity_details.num_issues} triangles with issues ({mesh_integrity_details.num_issues/mesh_integrity_details.num_polygons*100:.1f}%)")
            logging.info(f"Issue area: {mesh_integrity_details.issue_area:.4f} m^2 ({mesh_integrity_details.issue_area/mesh_integrity_details.total_area*100:.2f}%)")
            
            logging.info(f"Mesh analysis completed for {mesh_file}, Issues found: {has_issues}")
            
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

if __name__=="__main__":
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled", "experiment_5"))
    
    logging.basicConfig(
        filename=os.path.join(DATASET_PATH, "mesh_analysis.log"),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    db_main_frame, db_results_frame = load_data_frames(DATASET_PATH)
    print(f"Loaded lab database with {len(db_main_frame)} entries.")
    
    dataset_mesh_path = os.path.join(DATASET_PATH, "main")
    process_mesh_files(dataset_mesh_path, db_results_frame)

    update_data_frames(DATASET_PATH, db_main_frame, db_results_frame)