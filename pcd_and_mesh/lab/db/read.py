"""
This script contains functions to read the lab collected data.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from schema import AttributeSchema, ResultColumns
else:
    if __package__ is None or __package__ == "":
        # Assuming running as a script from the parent directory
        from db.schema import AttributeSchema, ResultColumns
    else:
        from .schema import AttributeSchema, ResultColumns

def load_data_frames(dataset_path: str, *,
        expected_columns: set = None, expected_result_columns: set = None,
        main_file_name: str = "lab_db.csv", 
        results_file_name: str = "lab_db_results.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the lab database from CSV files into a pandas DataFrame.
    Currently, there are two filesL: one for main attributes, and one for results.
    
    expected_columns = set(AttributeSchema.Columns.__members__.values())
    expected_result_columns = set(ResultColumns.__members__.values())
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Database file not found: {dataset_path}")
    
    if expected_columns is None:
        expected_columns = set(AttributeSchema.Columns.__members__.values())
    if expected_result_columns is None:
        expected_result_columns = set(ResultColumns.__members__.values())

    db_main_path = os.path.join(dataset_path, main_file_name)
    if not os.path.exists(db_main_path):
        raise FileNotFoundError(f"Database CSV file not found: {db_main_path}")
    df_main = pd.read_csv(db_main_path)
    missing_columns = expected_columns - set(df_main.columns)
    if missing_columns:
        raise ValueError(f"Database is missing expected columns: {missing_columns}")
    
    db_results_path = os.path.join(dataset_path, results_file_name)
    if not os.path.exists(db_results_path):
        raise FileNotFoundError(f"Database results CSV file not found: {db_results_path}")
    df_results = pd.read_csv(db_results_path)
    missing_result_columns = expected_result_columns - set(df_results.columns)
    if missing_result_columns:
        raise ValueError(f"Database results is missing expected columns: {missing_result_columns}")
    
    return df_main, df_results

def check_mesh_crop(results_frame: pd.DataFrame, results_index: int,
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

def check_pcd_crop(results_frame: pd.DataFrame, results_index: int,
    dataset_path: str, row_label: str, *, 
    pcd_dir: str = 'pcd', output_dir: str = 'pcd_cropped', repeat_crop: bool = False) -> bool:
    """
    Checks if the point cloud for the given row_label has already been cropped and saved.
    """
    output_path = os.path.join(dataset_path, output_dir)
    if os.path.exists(output_path):
        pcd_file = os.path.join(output_path, f"{row_label}.ply")
        if os.path.exists(pcd_file):
            return True
        else:
            print(f"PCD file not found: {pcd_file}.")
    else:
        print(f"Output directory not found: {output_path}.")
    return False