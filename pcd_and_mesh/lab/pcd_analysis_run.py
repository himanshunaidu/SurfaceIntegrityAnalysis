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
import optuna
from optuna import Trial
import warnings

from vision_utils.pcd import check_integrity as check_pcd_integrity, IntegrityDetails as PCDIntegrityDetails
from utils.plane import get_plane_mesh, get_viz_with_transparency
from utils.stats import get_array_stats
from db.schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
from db.read import load_data_frames
from db.update import update_data_frames, update_results_data_frame
from db.calc import calc_integrity_issue

def filter_row(
    main_row: pd.Series,
    results_row: pd.Series, *,
    pcd_file: str
    ) -> bool:
    """
    An ad-hoc function to filter rows that need to be processed.
    """
    # For now, we only process subdirectories that had board placement down
    pcd_file_name = os.path.splitext(os.path.basename(pcd_file))[0]
    
    board_placement = int(pcd_file_name.split("-")[3])
    print(f"Subdirectory {pcd_file_name} has board_placement = {board_placement}")
    return board_placement == 1
    # return True

def process_pcd_files(dataset_path: str, db_main_frame: pd.DataFrame, db_results_frame: pd.DataFrame, *, 
                    pcd_dir: str = 'pcd_cropped', output_dir: str = 'pcd_analysis',
                    depth_thr: float = 0.004, near_plane_thr: float = 0.05, 
                    min_angle_thr: float = 15.0,
                    dbscan_eps: float = 0.05, dbscan_min_samples: int = 5,
                    normal_radius: float = 0.03, normal_knn: int = 30, normal_orient_k: int = 50,
                    boundary_radius: float = 0.05, boundary_knn: int = 30, boundary_angle_thr: float = 60.0,
                    issue_percent_thr: float = 0.005) -> None:
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
        trial_name = row[AttributeSchema.Columns.TRIAL_NAME.value]
        
        num_issues = 0
        row_pcds = [f for f in point_cloud_file_names if f.startswith(trial_name)]
        if not row_pcds:
            # print(f"No point cloud file found for trial_name '{trial_name}' in row index {index}")
            continue
        
        for pcd_file in row_pcds:
            pcd_file_path = os.path.join(dataset_path, pcd_dir, pcd_file)
            
            if not filter_row(db_main_frame.iloc[index], row, pcd_file=pcd_file):
                continue
            
            analyzed_pcd, has_issues, pcd_integrity_details = check_pcd_integrity(
                pcd_file_path, depth_thr=depth_thr, near_plane_thr=near_plane_thr,
                min_angle_thr=min_angle_thr, dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
                normal_radius=normal_radius, normal_knn=normal_knn, normal_orient_k=normal_orient_k,
                boundary_radius=boundary_radius, boundary_knn=boundary_knn, boundary_angle_thr=boundary_angle_thr,
                issue_percent_thr=issue_percent_thr
            )
            
            output_pcd_file_path = os.path.join(output_dir_path, pcd_file)
            
            if has_issues: num_issues += 1
        db_results_frame.at[index, ResultColumns.POINT_CLOUD_RESULT.value] = num_issues
        print(f"Completed processing for row index {index}, trial_name '{trial_name}': point_cloud_result = {num_issues}")

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
    
    POINT_CLOUD_RESULT_COL = ResultColumns.POINT_CLOUD_RESULT.value
    POINT_CLOUD_NUMBER_COL = ResultColumns.POINT_CLOUD_NUMBER.value
    
    combined_df = pd.DataFrame(columns=[GAP_WIDTH_COL, GAP_DEPTH_COL, SURFACE_HEIGHT_DIFFERENCE_COL, BOARD_PLACEMENT_COL, POINT_CLOUD_RESULT_COL, POINT_CLOUD_NUMBER_COL])
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
                POINT_CLOUD_RESULT_COL: result_row[POINT_CLOUD_RESULT_COL] if not pd.isna(result_row[POINT_CLOUD_RESULT_COL]) else 0,
                POINT_CLOUD_NUMBER_COL: result_row[POINT_CLOUD_NUMBER_COL] if not pd.isna(result_row[POINT_CLOUD_NUMBER_COL]) else 0
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
                combined_df.at[idx, POINT_CLOUD_RESULT_COL] += combined_row[POINT_CLOUD_RESULT_COL]
                combined_df.at[idx, POINT_CLOUD_NUMBER_COL] += combined_row[POINT_CLOUD_NUMBER_COL]
            else:
                combined_row_df = pd.DataFrame([combined_row])
                combined_df = pd.concat([combined_df, combined_row_df], ignore_index=True)
    
    return combined_df

def loss(combined_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Analyzes the combined results DataFrame and calculates the loss.
    This loss is based on a kind of cross-entropy between actual and ideal results.
    
    Also returns total values, false positive rate and false negative rate.
    """
    POINT_CLOUD_RESULT_COL = ResultColumns.POINT_CLOUD_RESULT.value
    POINT_CLOUD_NUMBER_COL = ResultColumns.POINT_CLOUD_NUMBER.value
    
    total = 0
    false_total, false_positive = 0, 0
    true_total, false_negative = 0, 0
    
    loss_value = 0.0
    fpr, fnr = 0.0, 0.0
    
    for index, row in enumerate(combined_df.itertuples(index=False)):        
        y_total = getattr(row, POINT_CLOUD_NUMBER_COL)
        y_true = calc_integrity_issue(combined_df, index) * y_total
        if y_total == 0:
            continue
        y_pred = getattr(row, POINT_CLOUD_RESULT_COL)
        
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

def objective(trial: Trial = None):
    depth_thr = trial.suggest_categorical("depth_thr", [0.001, 0.005, 0.01, 0.02, 0.05])
    near_plane_thr = trial.suggest_categorical("near_plane_thr", [0.05])
    min_angle_thr = trial.suggest_discrete_uniform("min_angle_thr", 2.5, 10, 2.5)
    dbscan_eps = trial.suggest_categorical("dbscan_eps", [0.03, 0.05, 0.1])
    dbscan_min_samples = trial.suggest_categorical("dbscan_min_samples", [3, 5, 10])
    issue_percent_thr = trial.suggest_categorical("issue_percent_thr", [0.001, 0.01, 0.05, 0.1])
    
    logging.info(f"Trial parameters: depth_thr={depth_thr}, near_plane_thr={near_plane_thr}, min_angle_thr={min_angle_thr}, dbscan_eps={dbscan_eps}, dbscan_min_samples={dbscan_min_samples}, issue_percent_thr={issue_percent_thr}\n")
    
    DATASET_MAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled"))
    DATASETS = ["experiment_4", "experiment_5", "experiment_6"]
    
    dataset_dfs = []
    
    for dataset in DATASETS:
        DATASET_PATH = os.path.join(DATASET_MAIN_PATH, dataset)
        logging.info(f"Starting trial for dataset: {DATASET_PATH}")
        db_main_frame, db_results_frame = load_data_frames(DATASET_PATH)
        dataset_pcd_path = DATASET_PATH #os.path.join(DATASET_PATH, "main")
        process_pcd_files(
            dataset_pcd_path, db_main_frame, db_results_frame,
            depth_thr=depth_thr, near_plane_thr=near_plane_thr, min_angle_thr=min_angle_thr,
            dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
            issue_percent_thr=issue_percent_thr
        )   
        num_rows_with_issues = db_results_frame[db_results_frame[ResultColumns.POINT_CLOUD_RESULT.value] > 0].shape[0]
        logging.info(f"Trial completed. Number of rows with point cloud issues: {num_rows_with_issues}\n")
        
        dataset_dfs.append((db_main_frame, db_results_frame))
    
    combined_df = process_results(dataset_dfs)
    loss_value, total, fpr, fnr = loss(combined_df)
    logging.info(f"Combined results loss: {loss_value}, Total: {total}, FPR: {fpr}, FNR: {fnr}\n")
    return loss_value

if __name__=="__main__":
    DATASET_MAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled"))
    logging.basicConfig(
        filename=os.path.join(DATASET_MAIN_PATH, "pcd_analysis_run_board_placement_1.log"),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    study = optuna.create_study(direction="minimize", study_name="pcd_integrity_optimization")
    study.optimize(objective, n_trials=500, timeout=14400)  # 4 hours timeout
    
    # Print the best trial
    best_trial = study.best_trial
    logging.info("Best trial:")
    logging.info(f"Best trial parameters: {best_trial.params}")
    logging.info(f"Best trial value (loss): {best_trial.value}")