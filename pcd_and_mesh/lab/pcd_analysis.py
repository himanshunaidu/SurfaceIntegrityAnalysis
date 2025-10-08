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

from vision_utils.pcd import check_integrity as check_pcd_integrity, IntegrityDetails as PCDIntegrityDetails
from utils.plane import get_plane_mesh, get_viz_with_transparency
from utils.stats import get_array_stats
from db.schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
from db.read import load_data_frames
from db.update import update_data_frames, update_results_data_frame

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
            
            analyzed_pcd, has_issues, pcd_integrity_details = check_pcd_integrity(pcd_file_path)
            logging.info(f"Signed distance stats: {get_array_stats(pcd_integrity_details.distance_arr)}")
            logging.info(f"Angle stats: {get_array_stats(pcd_integrity_details.angle_arr)}")
            logging.info(f"Angle median = {pcd_integrity_details.angle_median:.2f} deg, MAD = {pcd_integrity_details.angle_mad:.2f} deg")
            logging.info(f"Angle threshold = {pcd_integrity_details.angle_threshold} deg")
            logging.info(f"Depth threshold = {0.004} m") # TODO: Remove hardcoding
            logging.info(f"Issue points: {pcd_integrity_details.num_issues}/{pcd_integrity_details.num_points}")
            
            logging.info(f"Point cloud analysis completed. Issues detected: {has_issues}")
            
            output_pcd_file_path = os.path.join(output_dir_path, pcd_file)
            o3d.io.write_point_cloud(output_pcd_file_path, analyzed_pcd)
            print(f"Saved analyzed point cloud to: {output_pcd_file_path}")
            
            if has_issues: num_issues += 1
        db_results_frame.at[index, ResultColumns.POINT_CLOUD_RESULT.value] = num_issues
        print(f"Updated DataFrame for row index {index}, trial_name '{trial_name}': point_cloud_result = {num_issues}")
    
if __name__=="__main__":
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled", "experiment_4"))
    
    logging.basicConfig(
        filename=os.path.join(DATASET_PATH, "pcd_analysis.log"),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    db_main_frame, db_results_frame = load_data_frames(DATASET_PATH)
    print(f"Loaded lab database with {len(db_main_frame)} entries.")
    
    process_pcd_files(DATASET_PATH, db_results_frame)
    
    update_data_frames(DATASET_PATH, db_main_frame, db_results_frame)