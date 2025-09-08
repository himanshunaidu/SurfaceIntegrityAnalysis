# Baseline pipeline: moment features + simple univariate tests (no Gaussian assumptions)
# ---------------------------------------------------------------------------------
# This notebook-style cell defines a tiny API you can drop your data into.
# It computes per-surface moments (mean, std, skew, kurtosis) for the angles
# and runs Mann–Whitney U tests between unbroken (0) and broken (1) surfaces,
# plus Benjamini–Hochberg FDR correction and Cliff's delta effect size.
#
# Notes:
# - We center angles via circular mean and wrap to (-pi, pi] before computing linear moments,
#   which keeps this "simple" yet robust for concentrated angle clouds.
# - Tests use Mann–Whitney U (two-sided). Effect = Cliff's delta.

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, mannwhitneyu
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from PIL import Image

from utils.surface_normals import get_segmentation_mask, compute_surface_normals, get_normal_angles, \
    visualize_normals_on_image, plot_histogram_with_image, visualize_surface_integrity


# -----------------------------
# Utility: circular centering
# -----------------------------
def wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def center_angles(angles: np.ndarray) -> np.ndarray:
    """Center angles to have circular mean ≈ 0 and wrap to (-pi, pi]."""
    angles = wrap_pi(angles)
    # circular mean
    mu = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    centered = wrap_pi(angles - mu)
    return centered


# -----------------------------
# Per-surface moment features
# -----------------------------
def moment_features(angles: np.ndarray) -> Dict[str, float]:
    """Return simple linear moments after circular centering."""
    a = center_angles(np.asarray(angles))
    # Linear moments on centered angles (simple baseline)
    feats = {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else np.nan,
        "skew": float(skew(a, bias=False)) if a.size > 2 else np.nan,
        "kurt": float(kurtosis(a, fisher=True, bias=False)) if a.size > 3 else np.nan,
    }
    return feats


def extract_per_surface_features(
    normals: List[np.ndarray]
) -> pd.DataFrame:
    """Compute moment features per surface, for AZ and optionally EL."""
    rows = []
    n = len(normals)
    for i in range(n):
        row = {"surface_id": i}
        normal_feats = moment_features(normals[i])
        for k, v in normal_feats.items():
            row[f"normal_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Stats: Mann–Whitney + BH-FDR
# -----------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size (x vs y)."""
    x = np.asarray(x)
    y = np.asarray(y)
    diffs = np.subtract.outer(x, y)
    n_greater = np.sum(diffs > 0)
    n_less = np.sum(diffs < 0)
    return (n_greater - n_less) / (x.size * y.size)


def mannwhitney_tests(df: pd.DataFrame, labels: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
    """Run two-sided Mann–Whitney U for each feature; add BH-FDR q-values and effect sizes."""
    labels = np.asarray(labels).astype(int)
    g0 = df.loc[labels == 0]
    g1 = df.loc[labels == 1]
    results = []
    for feat in feature_cols:
        x = g0[feat].dropna().values
        y = g1[feat].dropna().values
        if x.size < 1 or y.size < 1:
            p = np.nan
            U = np.nan
            delta = np.nan
            med0 = np.nan
            med1 = np.nan
            mean0 = np.nan
            mean1 = np.nan
        else:
            res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
            U = float(res.statistic)
            p = float(res.pvalue)
            delta = float(cliffs_delta(x, y))
            med0, med1 = np.median(x), np.median(y)
            mean0, mean1 = np.mean(x), np.mean(y)
        results.append(
            dict(
                feature=feat,
                n0=int((labels == 0).sum()),
                n1=int((labels == 1).sum()),
                median_unbroken=float(med0) if not np.isnan(med0) else np.nan,
                median_broken=float(med1) if not np.isnan(med1) else np.nan,
                mean_unbroken=float(mean0) if not np.isnan(mean0) else np.nan,
                mean_broken=float(mean1) if not np.isnan(mean1) else np.nan,
                cliffs_delta=delta,
                U_stat=U,
                p_value=p,
            )
        )
    out = pd.DataFrame(results).sort_values("p_value", na_position="last")
    # Benjamini–Hochberg FDR
    m = out["p_value"].notna().sum()
    if m > 0:
        pvals = out["p_value"].values
        ranks = np.argsort(np.argsort(pvals, kind="mergesort"), kind="mergesort") + 1  # 1..m, stable
        q = np.full_like(pvals, np.nan, dtype=float)
        mask = ~np.isnan(pvals)
        p_sorted_idx = np.argsort(pvals[mask], kind="mergesort")
        p_sorted = pvals[mask][p_sorted_idx]
        q_sorted = np.minimum.accumulate((p_sorted * m / (np.arange(1, np.sum(mask) + 1)))[::-1])[::-1]
        q_vals = np.full_like(pvals, np.nan, dtype=float)
        q_vals[mask][p_sorted_idx] = q_sorted
        out["q_value_BH"] = q_vals
    else:
        out["q_value_BH"] = np.nan
    return out

def process_dataset(dataset_csv_path, segmentation_class_ids, bounds, intrinsics, step=2, *,
                    surface_integrity_col='sidewalk_surface_integrity'):
    """
    Process the dataset to compute surface normal metrics for each frame.
    """
    df = pd.read_csv(dataset_csv_path)
    normals = []
    labels = []

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
        normal = compute_surface_normals(
            depth, fx, fy, cx, cy, step=step, segmentation_mask=filtered_segmentation_mask)
        normal_angles = get_normal_angles(normal)

        normals.append(normal_angles)
        surface_integrity = row[surface_integrity_col] if surface_integrity_col in row else 'Correct'
        label = ''
        if surface_integrity == 'Correct' or surface_integrity == 'Occluded' or surface_integrity == 'Not Sure':
            label = 0
        else:
            label = 1
        labels.append(label)

    return normals, labels

# -----------------------------
# Demo with simple synthetic data
# -----------------------------
# Replace this block with your real data.
DATASET_PATH = 'dataset/ios_point_mapper/'
DATASET_CSV_PATH = 'dataset/ios_point_mapper/dataset_surface_integrity.csv'
DATASET_COLS = [
    'rgb_frame_path',
    'depth_frame_path',
    'annotation_frame_path',
    'odometry_timestamp',
    'location_timestamp'
]
SURFACE_INTEGRITY_COL = 'sidewalk_surface_integrity'

if __name__=='__main__':
    segmentation_class_ids = [22, 9, 25]  # sidewalk, curb ramp, tactile paving
    bounds = (0.0, 0.5, 1.0, 0.9)  # Example bounds
    fx, fy = 1335.0, 1335.0
    cx, cy = 960.0, 720.0
    intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    
    normals, labels = process_dataset(
        DATASET_CSV_PATH, segmentation_class_ids, bounds, intrinsics, step=2
    )
    
    per_surface = extract_per_surface_features(normals)
    per_surface["label"] = labels
    
    # Select feature columns programmatically
    feature_cols = [c for c in per_surface.columns if c not in ("surface_id", "label")]
    
    test_results = mannwhitney_tests(per_surface, labels, feature_cols)
    print("Per-surface features:")
    print(per_surface.head())
    
    print("\nMann–Whitney U test results:")
    print(test_results)

# np.random.seed(7)
# n_surfaces = 30
# normals = []
# labels = np.zeros(n_surfaces, dtype=int)

# for i in range(n_surfaces):
#     if i < n_surfaces // 2:
#         # unbroken: concentrated around 0 rad
#         normal = np.random.vonmises(mu=0.0, kappa=20, size=600)
#         labels[i] = 0
#     else:
#         # broken: more spread and a bit heavier tails
#         normal = np.random.vonmises(mu=0.0, kappa=5, size=600)
#         # add a small secondary mode to emulate irregularity
#         idx = np.random.choice(len(normal), size=len(normal)//6, replace=False)
#         normal[idx] += np.random.vonmises(mu=np.deg2rad(25), kappa=15, size=len(idx))
#         labels[i] = 1
#     normals.append(normal)

# # -----------------------------
# # Run the pipeline
# # -----------------------------
# per_surface = extract_per_surface_features(normals)
# per_surface["label"] = labels

# # Select feature columns programmatically
# feature_cols = [c for c in per_surface.columns if c not in ("surface_id", "label")]
# test_results = mannwhitney_tests(per_surface, labels, feature_cols)

# print("Per-surface features:")
# print(per_surface.head())

# print("\nMann–Whitney U test results:")
# print(test_results)