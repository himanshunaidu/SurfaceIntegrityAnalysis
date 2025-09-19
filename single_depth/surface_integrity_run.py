import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.surface_normals import get_segmentation_mask, compute_surface_normals, get_normal_angles, \
    visualize_normals_on_image, plot_histogram_with_image, visualize_surface_integrity

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'broken_sidewalk_2'))
# DATASET_CSV_PATH = 'dataset/dataset_surface_integrity.csv'
DATASET_CSV_PATH = os.path.join(DATASET_PATH, 'dataset_subset.csv')
DATASET_COLS = [
    'rgb_frame_path',
    'depth_frame_path',
    'annotation_frame_path',
    'odometry_timestamp',
    'location_timestamp'
]

HISTOGRAM_PATH = os.path.join(DATASET_PATH, 'histograms')
HISTOGRAM_PERCENTILES = [40, 50, 60, 70, 80]
if not os.path.exists(HISTOGRAM_PATH):
    os.makedirs(HISTOGRAM_PATH)

def compute_surface_normal_metrics(normal_angles):
    """
    Compute metrics such as mean, standard deviation, skewness, and kurtosis from the normal angles.
    """
    mean = np.mean(normal_angles)
    std = np.std(normal_angles)
    
    outliers = np.abs(normal_angles - mean) > 3 * std
    # normal_angles = normal_angles[outliers]

    metrics = {
        "mean": mean,
        "std": std,
        "skewness": abs(pd.Series(normal_angles).skew()),
        "kurtosis": pd.Series(normal_angles).kurtosis(),
        "outliers_count": np.sum(outliers),
        "count": len(normal_angles)
    }
    return metrics

def create_histogram(normal_angles, bins=50):
    """
    Create a histogram of the normal angles.
    """
    valid_angles = normal_angles[np.isfinite(normal_angles)]
    hist, bin_edges = np.histogram(valid_angles, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return hist, bin_centers

def process_dataset(dataset_csv_path, segmentation_class_ids, bounds, intrinsics, step=2):
    """
    Process the dataset to compute surface normal metrics for each frame.
    """
    df = pd.read_csv(dataset_csv_path)
    results = []

    for index, row in tqdm(df.iterrows(), desc="Processing dataset", total=len(df)):
        rgb_frame_path = os.path.join(DATASET_PATH, row['rgb_frame_path'].lstrip('/'))
        depth_frame_path = os.path.join(DATASET_PATH, row['depth_frame_path'].lstrip('/'))
        annotation_frame_path = os.path.join(DATASET_PATH, row['annotation_frame_path'].lstrip('/'))

        # Load and preprocess the frames
        rgb = Image.open(rgb_frame_path)
        depth = Image.open(depth_frame_path)
        segmentation = Image.open(annotation_frame_path)

        rgb = rgb.resize(depth.size, Image.BILINEAR)
        segmentation = segmentation.resize(depth.size, Image.NEAREST)
        depth = np.array(depth)
        segmentation = np.array(segmentation)

        # Compute surface normals
        filtered_segmentation_mask = get_segmentation_mask(segmentation, segmentation_class_ids, bounds)
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        normals = compute_surface_normals(
            depth, fx, fy, cx, cy, step=step, segmentation_mask=filtered_segmentation_mask)
        normal_angles = get_normal_angles(normals)
        
        # normal_image = visualize_normals_on_image(np.array(rgb), normals, step=step, scale=step*2,
        #                                           normal_angles=normal_angles, 
        #                                           percentiles=HISTOGRAM_PERCENTILES)
        normal_image = visualize_surface_integrity(rgb=np.array(rgb), normals=normals, step=step, scale=step*2)
        plot_histogram_with_image(normal_image, normal_angles, title="Surface Normals Histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(HISTOGRAM_PATH, f"{row['rgb_frame_path'].split('/')[-1].replace('.png', '')}_histogram.png"))
        plt.clf()
        plt.close()
        
        metrics = compute_surface_normal_metrics(normal_angles)
        
        # metrics['rgb_frame_path'] = row['rgb_frame_path']
        metrics['sidewalk_surface_integrity'] = row.get('sidewalk_surface_integrity', 'Not Sure')
        metrics['normal_angles'] = normal_angles
        results.append(metrics)

    return pd.DataFrame(results)

def post_hoc_analysis(metrics_df):
    """
    Perform post-hoc analysis on the computed metrics.
    """
    pass
    

if __name__ == "__main__":
    # Example usage
    segmentation_class_ids = [22, 9, 25]  # sidewalk, curb ramp, tactile paving
    bounds = (0.0, 0.5, 1.0, 0.9)  # Example bounds
    fx, fy = 1335.0, 1335.0
    cx, cy = 960.0, 720.0
    intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

    metrics_df = process_dataset(DATASET_CSV_PATH, segmentation_class_ids, bounds, intrinsics, step=2)
    print(metrics_df.head())

    corrected_sidewalk_metrics_df = metrics_df[(metrics_df['sidewalk_surface_integrity'] == 'Correct') \
        | (metrics_df['sidewalk_surface_integrity'] == 'Curb Ramp') \
            | (metrics_df['sidewalk_surface_integrity'] == 'Occluded')]
    corrected_sidewalk_metrics = {
        'count': len(corrected_sidewalk_metrics_df),
        "min": corrected_sidewalk_metrics_df['mean'].min(),
        "max": corrected_sidewalk_metrics_df['mean'].max(),
        "mean": corrected_sidewalk_metrics_df['mean'].mean(),
        "std": corrected_sidewalk_metrics_df['std'].mean(),
        "skewness": corrected_sidewalk_metrics_df['skewness'].mean(),
        "kurtosis": corrected_sidewalk_metrics_df['kurtosis'].mean(),
        "outliers_proportion": corrected_sidewalk_metrics_df['outliers_count'].sum() / corrected_sidewalk_metrics_df['count'].sum()
    }
    corrected_sidewalk_normal_angles = corrected_sidewalk_metrics_df['normal_angles'].explode().dropna().astype(float).values
    corrected_sidewalk_hist, corrected_bin_centers = create_histogram(corrected_sidewalk_normal_angles)
    print("Corrected Sidewalk Histogram:", corrected_sidewalk_hist)
    print("Bin Centers:", corrected_bin_centers)

    incorrect_sidewalk_metrics_df = metrics_df[(metrics_df['sidewalk_surface_integrity'] == 'Broken') \
        | (metrics_df['sidewalk_surface_integrity'] == 'Gap')]
    incorrect_sidewalk_metrics = {
        'count': len(incorrect_sidewalk_metrics_df),
        "min": incorrect_sidewalk_metrics_df['mean'].min(),
        "max": incorrect_sidewalk_metrics_df['mean'].max(),
        "mean": incorrect_sidewalk_metrics_df['mean'].mean(),
        "std": incorrect_sidewalk_metrics_df['std'].mean(),
        "skewness": incorrect_sidewalk_metrics_df['skewness'].mean(),
        "kurtosis": incorrect_sidewalk_metrics_df['kurtosis'].mean(),
        "outliers_proportion": incorrect_sidewalk_metrics_df['outliers_count'].sum() / incorrect_sidewalk_metrics_df['count'].sum()
    }
    incorrect_sidewalk_normal_angles = incorrect_sidewalk_metrics_df['normal_angles'].explode().dropna().astype(float).values
    incorrect_sidewalk_hist, incorrect_bin_centers = create_histogram(incorrect_sidewalk_normal_angles)
    print("Incorrect Sidewalk Histogram:", incorrect_sidewalk_hist)
    print("Bin Centers:", incorrect_bin_centers)
    
    # Display histograms
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(corrected_bin_centers, corrected_sidewalk_hist, width=0.1, alpha=0.7, label='Corrected Sidewalk')
    plt.title('Corrected Sidewalk Normal Angles Histogram')
    plt.xlabel('Normal Angle (degrees)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.bar(incorrect_bin_centers, incorrect_sidewalk_hist, width=0.1, alpha=0.7, label='Incorrect Sidewalk', color='red')
    plt.title('Incorrect Sidewalk Normal Angles Histogram')
    plt.xlabel('Normal Angle (degrees)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()