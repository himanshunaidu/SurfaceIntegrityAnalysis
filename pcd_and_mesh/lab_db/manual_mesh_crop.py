"""
This script helps in processing all the meshes in the lab database, and setting them up for manual cropping.
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
import open3d as o3d

from schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS

def load_lab_db_frames(dataset_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the lab database from CSV files into a pandas DataFrame.
    Currently, there are two filesL: one for main attributes, and one for results.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Database file not found: {dataset_path}")

    db_main_path = os.path.join(dataset_path, "lab_db.csv")
    if not os.path.exists(db_main_path):
        raise FileNotFoundError(f"Database CSV file not found: {db_main_path}")
    df_main = pd.read_csv(db_main_path)
    expected_columns = set(AttributeSchema.Columns.__members__.values())
    missing_columns = expected_columns - set(df_main.columns)
    if missing_columns:
        raise ValueError(f"Database is missing expected columns: {missing_columns}")
    
    db_results_path = os.path.join(dataset_path, "lab_db_results.csv")
    if not os.path.exists(db_results_path):
        raise FileNotFoundError(f"Database results CSV file not found: {db_results_path}")
    df_results = pd.read_csv(db_results_path)
    expected_result_columns = set(ResultColumns.__members__.values())
    missing_result_columns = expected_result_columns - set(df_results.columns)
    if missing_result_columns:
        raise ValueError(f"Database results is missing expected columns: {missing_result_columns}")
    
    return df_main, df_results

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

def sync_lab_db_frames_with_mesh_files(dataset_path: str, db_results_frame: pd.DataFrame, *,
                    mesh_dir: str = 'mesh', output_dir: str = 'mesh_cropped') -> pd.DataFrame:
    """
    Syncs the lab database results frame with the actual mesh files present in the dataset path.
    """
    subtrial_directories = glob.glob(os.path.join(dataset_path, "*"))
    subtrial_directories.sort()
    subtrial_directory_names = [os.path.basename(d) for d in subtrial_directories if os.path.isdir(d)]
    # print(f"Found {len(subtrial_directory_names)} subtrial directories in {dataset_path}")
    for index, row in db_results_frame.iterrows():
        trial_name = row['trial_name']
        row_subtrials = [d for d in subtrial_directory_names if d.startswith(trial_name)]
        
        num_meshes = 0
        # if row_subtrials:
        # print(f"Found {len(row_subtrials)} subdirectories for trial_name '{trial_name}' in row index {index}")
        for sub_dir_name in row_subtrials:
            # Sync mesh cropping status for this subdirectory
            if check_mesh_crop(db_results_frame, index, dataset_path, sub_dir_name, mesh_dir=mesh_dir, output_dir=output_dir):
                num_meshes += 1

        # print(f"Completed syncing for row index {index}, trial_name '{trial_name}'. {len(row_subtrials)} subtrials processed.")
        # Update the dataframe with number of meshes found
        db_results_frame.at[index, ResultColumns.POLYGON_MESH_NUMBER.value] = num_meshes
        db_results_frame.at[index, ResultColumns.POLYGON_MESH_AVAILABLE.value] = (num_meshes > 0)
        print(f"Updated DataFrame for row '{trial_name}': polygon_mesh_number = {num_meshes}, polygon_mesh_available = {num_meshes > 0}")
    
    return db_results_frame

def check_mesh_crop(db_results_frame: pd.DataFrame, db_results_index: int,
    dataset_path: str, row_label: str, *, 
    mesh_dir: str = 'mesh', output_dir: str = 'mesh_cropped', repeat_crop: bool = True) -> bool:
    """
    Checks if there is a cropped mesh with the database results frame.
    """
    row_path = os.path.join(dataset_path, row_label)
    if not os.path.exists(row_path):
        raise FileNotFoundError(f"Row path not found: {row_path}")
    
    # Check if output directory exists and contains cropped mesh
    output_path = os.path.join(row_path, output_dir)
    if os.path.exists(output_path):
        mesh_cropped_files = glob.glob(os.path.join(output_path, "*.ply"))
        if mesh_cropped_files:
            return True
        else:
            print(f"No cropped mesh files found in {output_path}.")
    else:
        print(f"Output directory not found: {output_path}.")
    return False

def process_mesh_files(dataset_path: str, db_results_frame: pd.DataFrame, *, 
                    mesh_dir: str = 'mesh', output_dir: str = 'mesh_cropped') -> None:
    """
    Loads all mesh files from the dataset path based on the database results frame and opens them for manual cropping.
    
    Note: The db_results_frame contains column 'trial_name' based on which the directory structure is built (with some differences).
    Each row in db_results_frame corresponds to multiple subdirectories in dataset_path, and within those, the mesh files are located in mesh_dir.
        (Multiple subdirectories because of repeated trials with same parameters. We can call them subtrials.)

    The subtrials are named as per the 'trial_name' column in db_results_frame, along with 'a', 'b', ... suffixes for repeated trials.
    """
    subtrial_directories = glob.glob(os.path.join(dataset_path, "*"))
    subtrial_directories.sort()
    subtrial_directory_names = [os.path.basename(d) for d in subtrial_directories if os.path.isdir(d)]
    print(f"Found {len(subtrial_directory_names)} subtrial directories in {dataset_path}")
    for index, row in db_results_frame.iterrows():
        trial_name = row['trial_name']
        row_subtrials = [d for d in subtrial_directory_names if d.startswith(trial_name)]
        if not row_subtrials:
            # print(f"No subdirectory found for trial_name '{trial_name}' in row index {index}")
            continue
        print(f"Found {len(row_subtrials)} subdirectories for trial_name '{trial_name}' in row index {index}")

        for sub_dir_name in row_subtrials:
            # Setup mesh cropping for this subdirectory
            setup_mesh_crop(db_results_frame, index, dataset_path, sub_dir_name, mesh_dir=mesh_dir, output_dir=output_dir)
        
        print(f"Completed processing for row index {index}, trial_name '{trial_name}'. {len(row_subtrials)} subtrials processed.")

def setup_mesh_crop(db_results_frame: pd.DataFrame, db_results_index: int,
    dataset_path: str, row_label: str, *, 
    mesh_dir: str = 'mesh', output_dir: str = 'mesh_cropped', repeat_crop: bool = False) -> None:
    """
    Sets up the manual mesh cropping environment by getting the mesh, and opening the visualizer with editing capabilities.
    """
    row_path = os.path.join(dataset_path, row_label)
    if not os.path.exists(row_path):
        raise FileNotFoundError(f"Row path not found: {row_path}")
    
    # Create output directory if it doesn't exist
    output_path = os.path.join(row_path, output_dir)
    # But if it exists, contains a cropped mesh and repeat_crop is False, skip
    if os.path.exists(output_path) and not repeat_crop:
        mesh_cropped_files = glob.glob(os.path.join(output_path, "*.ply"))
        if mesh_cropped_files:
            print(f"Cropped mesh already exists in {output_path}. Skipping cropping for row '{row_label}'.")
            return
    os.makedirs(output_path, exist_ok=True)
    
    mesh_path = os.path.join(row_path, mesh_dir)
    if not os.path.exists(mesh_path):
        print(f"Mesh directory not found: {mesh_path}.")
        return
    
    mesh_files = glob.glob(os.path.join(mesh_path, "*.ply"))
    if not mesh_files:
        print(f"No PLY mesh files found in directory: {mesh_path}.")
        return
    
    mesh_file = mesh_files[0]
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    if mesh.is_empty():
        print(f"Loaded mesh is empty: {mesh_file}.")
        return
    
    print(f"Loaded mesh from {mesh_file} with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles for row '{row_label}'")
    
    mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()
    # mesh.paint_uniform_color([0.7, 0.7, 0.7])
    # Visualize the mesh with editing capabilities
    o3d.visualization.draw_geometries_with_editing([mesh])
    
    user_confirmation = input(f"Did you finish cropping the mesh for row '{row_label}'? (y/n): ").strip().lower()
    if user_confirmation != 'y':
        print("Mesh cropping not confirmed. Exiting without saving.")
        return
    
    # Update the dataframe to indicate cropping is done
    # Update polygon_mesh_available to True, polygon_mesh_number = polygon_mesh_number + 1
    # current_polygon_mesh_number = db_results_frame.at[db_results_index, ResultColumns.POLYGON_MESH_NUMBER.value]
    # if pd.isna(current_polygon_mesh_number):
    #     current_polygon_mesh_number = 0
    # db_results_frame.at[db_results_index, ResultColumns.POLYGON_MESH_NUMBER.value] = current_polygon_mesh_number + 1
    # db_results_frame.at[db_results_index, ResultColumns.POLYGON_MESH_AVAILABLE.value] = True
    # print(f"Updated DataFrame for row '{row_label}': polygon_mesh_number = {current_polygon_mesh_number + 1}, polygon_mesh_available = True")
    

if __name__=="__main__":
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled", "experiment_4"))
    
    db_main_frame, db_results_frame = load_lab_db_frames(DATASET_PATH)
    print(f"Loaded lab database with {len(db_main_frame)} entries.")

    dataset_mesh_path = os.path.join(DATASET_PATH, "main")
    db_results_frame = sync_lab_db_frames_with_mesh_files(dataset_mesh_path, db_results_frame)
    update_lab_db_frames(DATASET_PATH, db_main_frame, db_results_frame)
    
    db_main_frame, db_results_frame = load_lab_db_frames(DATASET_PATH)
    process_mesh_files(dataset_mesh_path, db_results_frame)
    sync_lab_db_frames_with_mesh_files(dataset_mesh_path, db_results_frame)