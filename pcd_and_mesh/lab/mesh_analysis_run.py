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
import optuna
from optuna import Trial

from vision_utils.mesh import check_integrity as check_mesh_integrity, IntegrityDetails as MeshIntegrityDetails
from utils.plane import get_plane_mesh, get_viz_with_transparency
from utils.stats import get_array_stats
from db.schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
from db.read import load_data_frames
from db.calc import calc_integrity_issue

def filter_row(
    results_row: pd.Series, *,
    sub_dir_name: str
    ) -> bool:
    """
    An ad-hoc function to filter rows that need to be processed.
    """
    # For now, we only process subdirectories that had board placement down
    board_placement = int(sub_dir_name.split("-")[3])
    print(f"Subdirectory {sub_dir_name} has board_placement = {board_placement}")
    return board_placement == 1
    # return True

def process_mesh_files(dataset_path: str, db_results_frame: pd.DataFrame, *, 
                    mesh_dir: str = 'mesh_cropped', output_dir: str = 'mesh_analyzed',
                    depth_thr: float = 0.005, near_plane_thr: float = 0.05, 
                    min_angle_thr: float = 10.0,
                    dbscan_eps: float = 0.05, dbscan_min_samples: int = 3,
                    issue_area_percent_thr: float = 0.005) -> None:
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
        trial_name = row[AttributeSchema.Columns.TRIAL_NAME.value]
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
            
            if not filter_row(row, sub_dir_name=sub_dir_name): 
                continue
            
            analyzed_mesh, has_issues, mesh_integrity_details = check_mesh_integrity(
                mesh_file, depth_thr=depth_thr, near_plane_thr=near_plane_thr, min_angle_thr=min_angle_thr,
                dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
                issue_area_percent_thr=issue_area_percent_thr)
            
            if has_issues: num_issues += 1
        db_results_frame.at[index, ResultColumns.POLYGON_MESH_RESULT.value] = num_issues
        print(f"Completed processing for row index {index}, trial_name '{trial_name}'. {num_issues} subtrials with issues.")

def process_results(dataset_dfs: list[tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """
    Processes the results from multiple datasets.
    For each row, gets the relevant attributes and results, and aggregates to similar trials in other datasets.
    """
    # Define the columns to be used for analysis
    GAP_WIDTH_COL = AttributeSchema.Columns.GAP_WIDTH.value
    GAP_DEPTH_COL = AttributeSchema.Columns.GAP_DEPTH.value
    SURFACE_HEIGHT_DIFFERENCE_COL = AttributeSchema.Columns.SURFACE_HEIGHT_DIFFERENCE.value
    BOARD_PLACEMENT_COL = AttributeSchema.Columns.BOARD_PLACEMENT.value
    
    POLYGON_MESH_RESULT_COL = ResultColumns.POLYGON_MESH_RESULT.value
    POLYGON_MESH_NUMBER_COL = ResultColumns.POLYGON_MESH_NUMBER.value
    
    # Combine all datasets into a single DataFrame for analysis
    combined_df = pd.DataFrame(columns=[GAP_WIDTH_COL, GAP_DEPTH_COL, SURFACE_HEIGHT_DIFFERENCE_COL, BOARD_PLACEMENT_COL, POLYGON_MESH_RESULT_COL, POLYGON_MESH_NUMBER_COL])
    for df_main, df_results in dataset_dfs:
        for row in df_main.itertuples(index=False):
            trial_name = getattr(row, AttributeSchema.Columns.TRIAL_NAME.value)
            matching_results = df_results[df_results[AttributeSchema.Columns.TRIAL_NAME.value] == trial_name]
            if matching_results.empty:
                continue
            result_row = matching_results.iloc[0]
            combined_row = {
                GAP_WIDTH_COL: getattr(row, GAP_WIDTH_COL),
                GAP_DEPTH_COL: getattr(row, GAP_DEPTH_COL),
                SURFACE_HEIGHT_DIFFERENCE_COL: getattr(row, SURFACE_HEIGHT_DIFFERENCE_COL),
                BOARD_PLACEMENT_COL: getattr(row, BOARD_PLACEMENT_COL),
                POLYGON_MESH_RESULT_COL: result_row[POLYGON_MESH_RESULT_COL] if not pd.isna(result_row[POLYGON_MESH_RESULT_COL]) else 0,
                POLYGON_MESH_NUMBER_COL: result_row[POLYGON_MESH_NUMBER_COL] if not pd.isna(result_row[POLYGON_MESH_NUMBER_COL]) else 0
            }
            # Check if exact row already exists
            existing_row = combined_df[
                (combined_df[GAP_WIDTH_COL] == combined_row[GAP_WIDTH_COL]) &
                (combined_df[GAP_DEPTH_COL] == combined_row[GAP_DEPTH_COL]) &
                (combined_df[SURFACE_HEIGHT_DIFFERENCE_COL] == combined_row[SURFACE_HEIGHT_DIFFERENCE_COL]) &
                (combined_df[BOARD_PLACEMENT_COL] == combined_row[BOARD_PLACEMENT_COL])
            ]
            if not existing_row.empty:
                idx = existing_row.index[0]
                combined_df.at[idx, POLYGON_MESH_RESULT_COL] += combined_row[POLYGON_MESH_RESULT_COL]
                combined_df.at[idx, POLYGON_MESH_NUMBER_COL] += combined_row[POLYGON_MESH_NUMBER_COL]
            else:
                combined_df = pd.concat([combined_df, pd.DataFrame([combined_row])], ignore_index=True)
    return combined_df

def loss(combined_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Analyzes the combined results DataFrame and calculates the loss.
    This loss is based on a kind of cross-entropy between actual and ideal results.
    
    Also returns total values, false positive rate and false negative rate.
    """    
    POLYGON_MESH_RESULT_COL = ResultColumns.POLYGON_MESH_RESULT.value
    POLYGON_MESH_NUMBER_COL = ResultColumns.POLYGON_MESH_NUMBER.value
    
    total = 0
    false_total, false_positive = 0, 0
    true_total, false_negative = 0, 0
    
    loss_value = 0.0
    fpr, fnr = 0.0, 0.0
    
    for index, row in enumerate(combined_df.itertuples(index=False)):        
        y_total = getattr(row, POLYGON_MESH_NUMBER_COL)
        y_true = calc_integrity_issue(combined_df, index) * y_total
        if y_total == 0:
            continue
        y_pred = getattr(row, POLYGON_MESH_RESULT_COL)
        
        p_true = y_true / y_total
        p_pred = y_pred / y_total
        p_pred = min(max(p_pred, 1e-6), 1 - 1e-6)  # Clamp to avoid log(0)
        loss_row = - (p_true * np.log(p_pred) + (1 - p_true) * np.log(1 - p_pred))
        loss_value += loss_row
        
        total += y_total
        if y_true == 0:
            false_total += y_total
            false_positive += (y_pred)
        else:
            true_total += y_total
            false_negative += (y_total - y_pred)
        
    fpr = (false_positive / false_total) if false_total > 0 else 0.0
    fnr = (false_negative / true_total) if true_total > 0 else 0.0
    return loss_value, total, fpr, fnr
    

def objective(trial: Trial):
    depth_thr = trial.suggest_categorical("depth_thr", [0.001, 0.005, 0.01, 0.02, 0.05])
    near_plane_thr = trial.suggest_categorical("near_plane_thr", [0.05])
    min_angle_thr = trial.suggest_discrete_uniform("min_angle_thr", 2.5, 10, 2.5)
    dbscan_eps = trial.suggest_categorical("dbscan_eps", [0.03, 0.05, 0.1])
    dbscan_min_samples = trial.suggest_categorical("dbscan_min_samples", [3, 5, 10])
    issue_area_percent_thr = trial.suggest_categorical("issue_area_percent_thr", [0.001, 0.01, 0.05, 0.1])
    
    logging.info(f"Starting trial with parameters: depth_thr={depth_thr}, near_plane_thr={near_plane_thr}, min_angle_thr={min_angle_thr}, dbscan_eps={dbscan_eps}, dbscan_min_samples={dbscan_min_samples}, issue_area_percent_thr={issue_area_percent_thr}\n")
    # logging.info(f"Trial parameters: depth_thr={depth_thr}, near_plane_thr={near_plane_thr}, min_angle_thr={min_angle_thr}, dbscan_eps={dbscan_eps}, dbscan_min_samples={dbscan_min_samples}, issue_area_percent_thr={issue_area_percent_thr}\n")
    
    DATASET_MAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled"))
    DATASETS = ["experiment_4", "experiment_5", "experiment_6"]
    
    dataset_dfs = []
    
    for dataset in DATASETS:
        DATASET_PATH = os.path.join(DATASET_MAIN_PATH, dataset)
        # logging.info(f"Starting trial for dataset: {DATASET_PATH}")
        db_main_frame, db_results_frame = load_data_frames(DATASET_PATH)
        dataset_mesh_path = os.path.join(DATASET_PATH, "main")
        process_mesh_files(
            dataset_mesh_path, db_results_frame,
            depth_thr=depth_thr, near_plane_thr=near_plane_thr, min_angle_thr=min_angle_thr,
            dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
            issue_area_percent_thr=issue_area_percent_thr
        )
        num_rows_with_issues = db_results_frame[db_results_frame[ResultColumns.POLYGON_MESH_RESULT.value] > 0].shape[0]
        # logging.info(f"Trial completed. Number of rows with polygon mesh issues: {num_rows_with_issues}\n")
        
        dataset_dfs.append((db_main_frame, db_results_frame))
    
    combined_df = process_results(dataset_dfs)
    loss_value, total, fpr, fnr = loss(combined_df)
    logging.info(f"Finished trial with parameters: depth_thr={depth_thr}, near_plane_thr={near_plane_thr}, min_angle_thr={min_angle_thr}, dbscan_eps={dbscan_eps}, dbscan_min_samples={dbscan_min_samples}, issue_area_percent_thr={issue_area_percent_thr}\n"+
        f"Combined results loss: {loss_value}, Total: {total}, FPR: {fpr}, FNR: {fnr}\n")
    return loss_value

if __name__=="__main__":
    DATASET_MAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled"))
    logging.basicConfig(
        filename=os.path.join(DATASET_MAIN_PATH, "mesh_analysis_run_board_placement_1_grid.log"),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # Set up grid sampler
    search_space = {
        "depth_thr": [0.001, 0.005, 0.01, 0.02, 0.05],
        "near_plane_thr": [0.05],
        "min_angle_thr": [2.5, 5.0, 7.5, 10.0],
        "dbscan_eps": [0.03, 0.05, 0.1],
        "dbscan_min_samples": [3, 5, 10],
        "issue_area_percent_thr": [0.001, 0.01, 0.05, 0.1]
    }
    sampler = optuna.samplers.GridSampler(search_space)
    
    study = optuna.create_study(direction="minimize", study_name="mesh_integrity_optimization", sampler=sampler)
    n_jobs=max(1, os.cpu_count()-2)
    study.optimize(objective, timeout=57600, n_jobs=n_jobs)  # 16 hours timeout
    
    # Print the best trial
    best_trial = study.best_trial
    logging.info("Best trial:")
    logging.info(f"Best trial parameters: {best_trial.params}")
    logging.info(f"Best trial value (loss): {best_trial.value}")