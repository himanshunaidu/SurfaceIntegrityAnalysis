import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

def get_segmentation_mask(segmentation, class_ids: list, bounds: tuple):
    min_x, min_y, max_x, max_y = bounds
    min_x = min_x * segmentation.shape[1]
    min_y = min_y * segmentation.shape[0]
    max_x = max_x * segmentation.shape[1]
    max_y = max_y * segmentation.shape[0]

    mask = np.zeros_like(segmentation, dtype=bool)
    for class_id in class_ids:
        mask |= (segmentation == class_id)

    mask &= (np.arange(segmentation.shape[0])[:, None] >= min_y) & \
             (np.arange(segmentation.shape[0])[:, None] < max_y) & \
             (np.arange(segmentation.shape[1])[None, :] >= min_x) & \
             (np.arange(segmentation.shape[1])[None, :] < max_x)

    # filtered = np.zeros_like(segmentation)
    # filtered[mask] = segmentation[mask]
    return mask

def compute_surface_normals(depth, fx, fy, cx, cy, step=20, segmentation_mask=None, segmentation_radius=10):
    h, w = depth.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)

    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            dz = depth[y, x]
            if dz == 0:
                continue
            
            if segmentation_mask is not None:
                seg = segmentation_mask[y, x]
                seg_dx = segmentation_mask[y, x + segmentation_radius] if x + segmentation_radius < w else segmentation_mask[y, x]
                seg_dy = segmentation_mask[y + segmentation_radius, x] if y + segmentation_radius < h else segmentation_mask[y, x]
                seg_dx_1 = segmentation_mask[y, x - segmentation_radius] if x - segmentation_radius >= 0 else segmentation_mask[y, x]
                seg_dy_1 = segmentation_mask[y - segmentation_radius, x] if y - segmentation_radius >= 0 else segmentation_mask[y, x]
                if not (seg and seg_dx and seg_dy and seg_dx_1 and seg_dy_1):
                    continue
            
            # Center point in 3D
            X = (x - cx) * dz / fx
            Y = (y - cy) * dz / fy
            P = np.array([X, Y, dz])

            # Neighbor in x-direction
            dz_dx = depth[y, x + 1]
            if dz_dx == 0:
                continue
            X_dx = (x + 1 - cx) * dz_dx / fx
            Y_dx = (y - cy) * dz_dx / fy
            P_dx = np.array([X_dx, Y_dx, dz_dx])

            # Neighbor in y-direction
            dz_dy = depth[y + 1, x]
            if dz_dy == 0:
                print(f"Skipping ({x}, {y}) due to zero depth in y-direction")
                continue
            X_dy = (x - cx) * dz_dy / fx
            Y_dy = (y + 1 - cy) * dz_dy / fy
            P_dy = np.array([X_dy, Y_dy, dz_dy])

            # Vectors and cross product
            v1 = P_dx - P
            v2 = P_dy - P
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm > 0:
                normals[y, x] = n / norm
                
            # print(f"Computed normal at ({x}, {y}): {P}, {P_dx}, {P_dy}")
            # print(f"Vectors: {v1}, {v2} -> Cross: {n}")
            # print(f"Norm (Normalized): {normals[y, x]}")
            # print(f"\n")

    return normals

def get_normal_angles(normals):
    valid_normals = normals[np.linalg.norm(normals, axis=2) > 0]
    if valid_normals.size == 0:
        return np.array([])

    # Compute angles with respect to the z-axis
    up = np.array([0, 0, 1])
    angles = np.arccos(np.clip(np.dot(valid_normals, up), -1.0, 1.0))
    
    valid_angles = angles[np.isfinite(angles)]
    if valid_angles.size == 0:
        return np.array([])
    return np.degrees(valid_angles)

def print_normals_statistics(normal_angles):
    valid_angles = normal_angles[np.isfinite(normal_angles)]
    if valid_angles.size == 0:
        print("No valid angles found.")
        return

    mean_angle = np.mean(valid_angles)
    std_angle = np.std(valid_angles)
    print(f"Mean Angle: {mean_angle}, Std Angle: {std_angle}")

def visualize_normals_on_image(rgb, normals, step=20, scale=20):
    vis = rgb.copy()
    for y in range(step, normals.shape[0] - step, step):
        for x in range(step, normals.shape[1] - step, step):
            n = normals[y, x]
            if np.linalg.norm(n) > 0:
                end_point = (int(x + scale * n[0]), int(y - scale * n[1]))
                cv2.arrowedLine(vis, (x, y), end_point, color=(0, 255, 0), thickness=scale//4, tipLength=scale/10)    
    return vis