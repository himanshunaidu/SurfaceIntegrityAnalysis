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