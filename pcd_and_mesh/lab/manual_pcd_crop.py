"""
This script helps in processing all the point clouds in the lab database, and setting them up for manual cropping.
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
import open3d as o3d

from db.schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
from db.read import load_data_frames
from db.update import update_data_frames, update_results_data_frame, sync_data_frames_with_pcd_files

def process_pcd_files(dataset_path: str, db_results_frame: pd.DataFrame, *, 
                    pcd_dir: str = 'pcd', output_dir: str = 'pcd_cropped') -> None:
    """
    Loads all point cloud files from the dataset path based on the database results frame and opens them for manual cropping.

    Note: The db_results_frame contains column 'trial_name' based on which the point cloud files are named (with some differences).
    Each row in db_results_frame corresponds to multiple point clouds in dataset_path.
        (Multiple point clouds because of repeated trials with same parameters. We can call them subtrials.)

    The subtrials are named as per the 'trial_name' column in db_results_frame, along with 'a', 'b', ... suffixes for repeated trials.
    """
    point_cloud_files = glob.glob(os.path.join(dataset_path, pcd_dir, "*.ply"))
    point_cloud_files.sort()
    point_cloud_file_names = [os.path.basename(f) for f in point_cloud_files]
    print(f"Found {len(point_cloud_file_names)} point cloud files in {os.path.join(dataset_path, pcd_dir)}")
    
    for index, row in db_results_frame.iterrows():
        trial_name = row['trial_name']
        row_pcds = [f for f in point_cloud_file_names if f.startswith(trial_name)]
        if not row_pcds:
            # print(f"No point cloud file found for trial_name '{trial_name}' in row index {index}")
            continue
        print(f"Found {len(row_pcds)} point cloud files for trial_name '{trial_name}' in row index {index}")

        for pcd_file in row_pcds:
            # Setup pcd cropping for this file
            setup_pcd_crop(db_results_frame, index, dataset_path, pcd_file[:-4], pcd_dir=pcd_dir, output_dir=output_dir)
            
            update_results_data_frame(db_results_frame, dataset_path)
        
        print(f"Completed processing for row index {index}, trial_name '{trial_name}'. {len(row_pcds)} subtrials processed.")
        update_results_data_frame(db_results_frame, dataset_path)

def setup_pcd_crop(db_results_frame: pd.DataFrame, db_results_index: int,
    dataset_path: str, row_label: str, *, 
    pcd_dir: str = 'pcd', output_dir: str = 'pcd_cropped', repeat_crop: bool = False) -> None:
    """
    Sets up the manual point cloud cropping environment by getting the point cloud, and opening the visualizer with editing capabilities.
    """    
    # Create output directory if it doesn't exist
    output_path = os.path.join(dataset_path, output_dir)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{row_label}.ply")
    if os.path.exists(output_file) and not repeat_crop:
        print(f"Output file already exists and repeat_crop is False: {output_file}. Skipping cropping.")
        return

    pcd_path = os.path.join(dataset_path, pcd_dir)
    if not os.path.exists(pcd_path):
        print(f"PCD directory not found: {pcd_path}.")
        return
    
    pcd_file = os.path.join(pcd_path, f"{row_label}.ply")
    if not os.path.exists(pcd_file):
        print(f"PCD file not found: {pcd_file}.")
        return
    
    pcd = o3d.io.read_point_cloud(pcd_file)
    if pcd.is_empty():
        print(f"Loaded point cloud is empty: {pcd_file}.")
        return

    print(f"Loaded point cloud from {pcd_file} with {len(pcd.points)} points for row '{row_label}'")

    # pcd.paint_uniform_color([0.7, 0.7, 0.7])
    # Visualize the point cloud with editing capabilities
    o3d.visualization.draw_geometries_with_editing([pcd])

    user_confirmation = input(f"Did you finish cropping the point cloud for row '{row_label}'? (y/n): ").strip().lower()
    if user_confirmation != 'y':
        print("Point cloud cropping not confirmed. Exiting without saving.")
        return
    
    # Update the dataframe to indicate cropping is done
    # Update point_cloud_available to True, point_cloud_number = point_cloud_number + 1
    current_point_cloud_number = db_results_frame.at[db_results_index, ResultColumns.POINT_CLOUD_NUMBER.value]
    if pd.isna(current_point_cloud_number):
        current_point_cloud_number = 0
    db_results_frame.at[db_results_index, ResultColumns.POINT_CLOUD_NUMBER.value] = current_point_cloud_number + 1
    db_results_frame.at[db_results_index, ResultColumns.POINT_CLOUD_AVAILABLE.value] = True
    print(f"Updated DataFrame for row '{row_label}': point_cloud_number = {current_point_cloud_number + 1}, point_cloud_available = True")

if __name__=="__main__":
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled", "experiment_4"))
    
    db_main_frame, db_results_frame = load_data_frames(DATASET_PATH)
    print(f"Loaded lab database with {len(db_main_frame)} entries.")
    
    db_results_frame = sync_data_frames_with_pcd_files(DATASET_PATH, db_results_frame)
    update_results_data_frame(db_results_frame, DATASET_PATH)

    # dataset_pcd_path = os.path.join(DATASET_PATH, "pcd")
    process_pcd_files(DATASET_PATH, db_results_frame)