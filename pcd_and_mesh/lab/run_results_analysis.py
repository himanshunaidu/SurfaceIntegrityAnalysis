"""
This script analyzes the results from the mesh analysis run on lab datasets.

It filters the log file to find the best performing parameter combinations based on the combined loss metric.
Saves the filtered results for further analysis.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import re

MAIN_DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled"))
RESULTS_FILE_PATH = "mesh_analysis_run_board_placement_1_grid.log"

OUTPUT_FILE_PATH = "filtered_mesh_analysis_results.csv"
OUTPUT_FILE_COLUMNS = [
    "id", "timestamp",
    "depth_thr", "near_plane_thr", "min_angle_thr", "dbscan_eps", "dbscan_min_samples", "issue_area_percent_thr",
    "combined_loss", "total", "fpr", "fnr"
]

# Start example: 
# 2025-10-20 12:26:39,042 - INFO - Starting trial with parameters: depth_thr=0.01, near_plane_thr=0.05, min_angle_thr=7.5, dbscan_eps=0.05, dbscan_min_samples=5, issue_area_percent_thr=0.01
TRIAL_START_PATTERN = r"^(.+?) - INFO - Starting trial with parameters: depth_thr=(.+?), near_plane_thr=(.+?), min_angle_thr=(.+?), dbscan_eps=(.+?), dbscan_min_samples=(.+?), issue_area_percent_thr=(.+)$"
# Finish example: 
# 2025-10-20 12:28:07,008 - INFO - Finished trial with parameters:...\n
# Combined results loss: 251.16409969314645, Total: 89.0, FPR: 0.0, FNR: 0.9736842105263158
TRIAL_FINISH_PATTERN = r"^(.+?) - INFO - Finished trial with parameters: depth_thr=(.+?), near_plane_thr=(.+?), min_angle_thr=(.+?), dbscan_eps=(.+?), dbscan_min_samples=(.+?), issue_area_percent_thr=(.+?)\nCombined results loss: (.+?), Total: (.+?), FPR: (.+?), FNR: (.+?)$"

output_df = pd.DataFrame(columns=OUTPUT_FILE_COLUMNS)

with open(os.path.join(MAIN_DATASET_PATH, RESULTS_FILE_PATH), 'r') as file:
    log_contents = file.read()
    
    # Get all trial finish lines
    # start_matches = re.findall(TRIAL_START_PATTERN, log_contents, re.MULTILINE)
    # print(f"Found {len(start_matches)} trial starts.")

    # Get all trial finish lines
    finish_matches = re.findall(TRIAL_FINISH_PATTERN, log_contents, re.MULTILINE)
    print(f"Found {len(finish_matches)} trial finishes.")
    
    for match in finish_matches:
        timestamp, depth_thr, near_plane_thr, min_angle_thr, dbscan_eps, dbscan_min_samples, issue_area_percent_thr, combined_loss, total, fpr, fnr = match
        print(f"Trial finished at {timestamp} with parameters:")
        print(f"  depth_thr: {depth_thr}, near_plane_thr: {near_plane_thr}, min_angle_thr: {min_angle_thr}, dbscan_eps: {dbscan_eps}, dbscan_min_samples: {dbscan_min_samples}, issue_area_percent_thr: {issue_area_percent_thr}")
        print(f"  Results - Combined Loss: {combined_loss}, Total: {total}, FPR: {fpr}, FNR: {fnr}")
        
        new_row = {
            "id": len(output_df) + 1,
            "timestamp": timestamp,
            "depth_thr": float(depth_thr),
            "near_plane_thr": float(near_plane_thr),
            "min_angle_thr": float(min_angle_thr),
            "dbscan_eps": float(dbscan_eps),
            "dbscan_min_samples": int(dbscan_min_samples),
            "issue_area_percent_thr": float(issue_area_percent_thr),
            "combined_loss": float(combined_loss),
            "total": float(total),
            "fpr": float(fpr),
            "fnr": float(fnr)
        }
        output_df.loc[len(output_df)] = new_row
        
# Save the filtered results
output_df.to_csv(os.path.join(MAIN_DATASET_PATH, OUTPUT_FILE_PATH), index=False)
print(f"Filtered results saved to {OUTPUT_FILE_PATH}")
