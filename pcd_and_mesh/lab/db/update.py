"""
This script contains functions to update the lab collected data.
"""
import os
import glob
import pandas as pd

from schema import AttributeSchema, ResultColumns
from read import check_mesh_crop, check_pcd_crop

def update_data_frames(
        dataset_path: str, main_frame: pd.DataFrame, results_frame: pd.DataFrame, *, 
        main_file_name: str = "lab_db.csv", 
        results_file_name: str = "lab_db_results.csv") -> None:
    """
    Updates the lab database CSV files from the given pandas DataFrames.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Database file not found: {dataset_path}")

    main_path = os.path.join(dataset_path, main_file_name)
    main_frame.to_csv(main_path, index=False)
    print(f"Updated main database CSV file: {main_path}")

    results_path = os.path.join(dataset_path, results_file_name)
    results_frame.to_csv(results_path, index=False)
    print(f"Updated results database CSV file: {results_path}")
    
def update_results_data_frame(results_frame: pd.DataFrame, dataset_path: str, *, results_file_name: str = 'lab_db_results.csv') -> None:
    """
    Saves the updated results DataFrame back to the CSV file.
    """
    out_path = os.path.join(dataset_path, results_file_name)
    results_frame.to_csv(out_path, index=False)
    print(f"Wrote updated results DataFrame with {len(results_frame)} entries to {out_path}")
    
def sync_data_frames_with_mesh_files(dataset_path: str, db_results_frame: pd.DataFrame, *,
                    mesh_dir: str = 'mesh', output_dir: str = 'mesh_cropped') -> pd.DataFrame:
    """
    Syncs the lab database results frame with the actual mesh files present in the dataset path.
    """
    subtrial_directories = glob.glob(os.path.join(dataset_path, "*"))
    subtrial_directories.sort()
    subtrial_directory_names = [os.path.basename(d) for d in subtrial_directories if os.path.isdir(d)]
    # print(f"Found {len(subtrial_directory_names)} subtrial directories in {dataset_path}")
    for index, row in db_results_frame.iterrows():
        trial_name = row['trial_name'] # TODO: Remove hardcoding
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

def sync_data_frames_with_pcd_files(dataset_path: str, db_results_frame: pd.DataFrame, *,
                    pcd_dir: str = 'pcd', output_dir: str = 'pcd_cropped')-> pd.DataFrame:
    point_cloud_files = glob.glob(os.path.join(dataset_path, pcd_dir, "*.ply"))
    point_cloud_files.sort()
    point_cloud_file_names = [os.path.basename(f) for f in point_cloud_files]
    print(f"Found {len(point_cloud_file_names)} point cloud files in {os.path.join(dataset_path, pcd_dir)}")
    
    for index, row in db_results_frame.iterrows():
        trial_name = row['trial_name']
        row_pcds = [f for f in point_cloud_file_names if f.startswith(trial_name)]
        
        num_trials = 0
        for pcd_file in row_pcds:
            if check_pcd_crop(db_results_frame, index, dataset_path, pcd_file[:-4], pcd_dir=pcd_dir, output_dir=output_dir):
                num_trials += 1
        
        # Update the dataframe with number of point clouds found
        db_results_frame.at[index, ResultColumns.POINT_CLOUD_NUMBER.value] = num_trials
        db_results_frame.at[index, ResultColumns.POINT_CLOUD_AVAILABLE.value] = (num_trials > 0)
        print(f"Updated DataFrame for row index {index}, trial_name '{trial_name}': point_cloud_number = {num_trials}, point_cloud_available = {num_trials > 0}")
        
    return db_results_frame