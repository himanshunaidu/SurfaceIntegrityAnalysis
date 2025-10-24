"""
This script analyses all the surface integrity results in the lab database.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from db.schema import AttributeSchema, ResultColumns, DatasetBuildPlan, DatasetBuildPlanOverrides, FACTORS
from db.read import load_data_frames

def process_results(dataset_dfs: list[tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """
    Processes the results from multiple datasets.
    For each row, gets the relevant attributes and results, and aggregates to similar trials in other datasets.
    """
    # Define the columns to be used for analysis
    X_COLUMN = AttributeSchema.Columns.GAP_WIDTH.value
    SIZE_COLUMN = AttributeSchema.Columns.GAP_DEPTH.value
    SUB_GROUP_COLUMN = AttributeSchema.Columns.SURFACE_HEIGHT_DIFFERENCE.value
    MAIN_GROUP_COLUMN = AttributeSchema.Columns.BOARD_PLACEMENT.value
    
    Y_NUMERATOR_COLUMN = ResultColumns.POLYGON_MESH_RESULT.value
    Y_DENOMINATOR_COLUMN = ResultColumns.POLYGON_MESH_NUMBER.value
    
    # Combine all datasets into a single DataFrame for analysis
    combined_df = pd.DataFrame(columns=[X_COLUMN, SIZE_COLUMN, SUB_GROUP_COLUMN, MAIN_GROUP_COLUMN, Y_NUMERATOR_COLUMN, Y_DENOMINATOR_COLUMN])
    for df_main, df_results in dataset_dfs:
        for row in df_main.itertuples(index=False):
            trial_name = getattr(row, AttributeSchema.Columns.TRIAL_NAME.value)
            matching_results = df_results[df_results[AttributeSchema.Columns.TRIAL_NAME.value] == trial_name]
            if matching_results.empty:
                continue
            result_row = matching_results.iloc[0]
            combined_row = {
                X_COLUMN: getattr(row, X_COLUMN),
                SIZE_COLUMN: getattr(row, SIZE_COLUMN),
                SUB_GROUP_COLUMN: getattr(row, SUB_GROUP_COLUMN),
                MAIN_GROUP_COLUMN: getattr(row, MAIN_GROUP_COLUMN),
                Y_NUMERATOR_COLUMN: result_row[Y_NUMERATOR_COLUMN] if not pd.isna(result_row[Y_NUMERATOR_COLUMN]) else 0,
                Y_DENOMINATOR_COLUMN: result_row[Y_DENOMINATOR_COLUMN] if not pd.isna(result_row[Y_DENOMINATOR_COLUMN]) else 0
            }
            # Check if exact row already exists
            existing_row = combined_df[
                (combined_df[X_COLUMN] == combined_row[X_COLUMN]) &
                (combined_df[SIZE_COLUMN] == combined_row[SIZE_COLUMN]) &
                (combined_df[SUB_GROUP_COLUMN] == combined_row[SUB_GROUP_COLUMN]) &
                (combined_df[MAIN_GROUP_COLUMN] == combined_row[MAIN_GROUP_COLUMN])
            ]
            if not existing_row.empty:
                idx = existing_row.index[0]
                combined_df.at[idx, Y_NUMERATOR_COLUMN] += combined_row[Y_NUMERATOR_COLUMN]
                combined_df.at[idx, Y_DENOMINATOR_COLUMN] += combined_row[Y_DENOMINATOR_COLUMN]
            else:
                combined_df = pd.concat([combined_df, pd.DataFrame([combined_row])], ignore_index=True)
    return combined_df

def analyze_results(combined_df: pd.DataFrame):
    """
    Analyzes the combined results DataFrame and generates plots.
    """
    X_COLUMN = AttributeSchema.Columns.GAP_WIDTH.value
    SIZE_COLUMN = AttributeSchema.Columns.GAP_DEPTH.value
    SUB_GROUP_COLUMN = AttributeSchema.Columns.SURFACE_HEIGHT_DIFFERENCE.value
    MAIN_GROUP_COLUMN = AttributeSchema.Columns.BOARD_PLACEMENT.value
    
    Y_NUMERATOR_COLUMN = ResultColumns.POLYGON_MESH_RESULT.value
    Y_DENOMINATOR_COLUMN = ResultColumns.POLYGON_MESH_NUMBER.value
    
    Y_VALUE_COLUMN = "Y_VALUE"
    SIZE_VALUE_COLUMN = "SIZE_VAL_COL"

    # Calculate the Y values as percentages
    combined_df[Y_VALUE_COLUMN] = combined_df.apply(
        lambda row: (row[Y_NUMERATOR_COLUMN] / row[Y_DENOMINATOR_COLUMN] * 100) if row[Y_DENOMINATOR_COLUMN] > 0 else 0,
        axis=1
    )
    # Filter out rows with zero denominator
    combined_df = combined_df[combined_df[Y_DENOMINATOR_COLUMN] > 0]
    
    # Temporary save for debugging
    combined_df.sort_values(by=[MAIN_GROUP_COLUMN, SUB_GROUP_COLUMN, SIZE_COLUMN, X_COLUMN], inplace=True)
    combined_df.to_csv("combined_results_debug.csv", index=False)
    
    if combined_df.empty:
        print("No valid data to analyze after filtering out zero denominators.")
        return
    
    # Assign size values for plotting
    unique_sizes = combined_df[SIZE_COLUMN].unique().tolist()
    unique_sizes.sort()
    combined_df[SIZE_VALUE_COLUMN] = ((combined_df[SIZE_COLUMN].apply(lambda x: unique_sizes.index(x)) + 1) * 10) ** 2
    print("Unique size values: ", unique_sizes, combined_df[SIZE_VALUE_COLUMN].unique().tolist())
    
    # Assign colors based on sub-group values
    colormap = cm.get_cmap('coolwarm', len(combined_df[SUB_GROUP_COLUMN].unique()))
    print("Colormap: ", colormap)
    unique_sub_groups = combined_df[SUB_GROUP_COLUMN].unique().tolist()
    unique_sub_groups.sort()
    color_mapping = {val: colormap(i) for i, val in enumerate(unique_sub_groups)}
    print("Color mapping: ", color_mapping)
    combined_df['Color'] = combined_df[SUB_GROUP_COLUMN].map(color_mapping)

    # Plotting
    # X_column will be the x-axis, Size_column will give markers of different sizes,
    # Sub_group_column will give different lines with different colors,
    # Main_group_column will give different sub-plots.
    fig, axs = plt.subplots(1, len(combined_df[MAIN_GROUP_COLUMN].unique()), figsize=(15, 5), sharey=True)
    if len(combined_df[MAIN_GROUP_COLUMN].unique()) == 1:
        axs = [axs]  # Ensure axs is always a list
    for ax, (main_group, group_df) in zip(axs, combined_df.groupby(MAIN_GROUP_COLUMN)):
        for groupby_tuple, sub_df in group_df.groupby([SUB_GROUP_COLUMN]):
            # Sort sub_df by X_COLUMN for consistent plotting
            sub_df = sub_df.sort_values(by=[X_COLUMN, SIZE_COLUMN])
            sub_group = groupby_tuple[0]
            ax.scatter(
                sub_df[X_COLUMN], sub_df[Y_VALUE_COLUMN], s=sub_df[SIZE_VALUE_COLUMN], 
                label=f"{SUB_GROUP_COLUMN}: {sub_group}", c=sub_df['Color']
            )
            ax.plot(sub_df[X_COLUMN], sub_df[Y_VALUE_COLUMN], linestyle='solid', alpha=0.5, color=color_mapping[sub_group])
        
        # Set up legend to show columns for color and size
        handles, labels = ax.get_legend_handles_labels()
        size_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"{SIZE_COLUMN}: {size}", 
                                   markerfacecolor='gray', markersize=(i + 1) * 10)
                        for i, size in enumerate(unique_sizes)]
        all_handles = handles + size_handles
        all_labels = labels + [f"{SIZE_COLUMN}: {size}" for size in unique_sizes]

        ax.set_title(f"{MAIN_GROUP_COLUMN}: {main_group}")
        ax.set_xlabel(X_COLUMN)
        ax.set_ylabel('Issue Percentage (%)')
        ax.legend(all_handles, all_labels)#, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    MAIN_DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled"))
    DATASETS = ["experiment_4", "experiment_5", "experiment_6"]
    
    dataset_dfs = []
    for dataset in DATASETS:
        dataset_path = os.path.join(MAIN_DATASET_PATH, dataset)
        print(f"Loading dataset from {dataset_path}...")
        df_main, df_results = load_data_frames(dataset_path)
        # print("Describe results DataFrame:")
        # print(df_results.dtypes)
        dataset_dfs.append((df_main, df_results))

    combined_df = process_results(dataset_dfs)
    analyze_results(combined_df)