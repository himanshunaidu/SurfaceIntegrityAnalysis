import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

DATASET_PATH = 'dataset/'
DATASET_CSV_PATH = 'dataset/dataset.csv'
DATASET_COLS = [
    'rgb_frame_path',
    'depth_frame_path',
    'annotation_frame_path',
    'odometry_timestamp',
    'location_timestamp'
]

HISTOGRAM_OUTPUT_PATH = 'dataset/histogram/'

def filter_segmentation(segmentation, class_id, bounds: tuple):
    min_x, min_y, max_x, max_y = bounds
    min_x = min_x * segmentation.shape[1]
    min_y = min_y * segmentation.shape[0]
    max_x = max_x * segmentation.shape[1]
    max_y = max_y * segmentation.shape[0]
    
    print(f"Filtering segmentation for class {class_id} within bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    
    mask = (segmentation == class_id) & \
        (np.arange(segmentation.shape[0])[:, None] >= min_y) & \
        (np.arange(segmentation.shape[0])[:, None] < max_y) & \
        (np.arange(segmentation.shape[1])[None, :] >= min_x) & \
        (np.arange(segmentation.shape[1])[None, :] < max_x)
    filtered = np.zeros_like(segmentation)
    filtered[mask] = segmentation[mask]
    return filtered

def compute_surface_normals(depth, fx, fy, cx, cy, step=20, segmentation=None, segmentation_radius=10):
    h, w = depth.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)

    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            dz = depth[y, x]
            if dz == 0:
                continue
            
            if segmentation is not None:
                seg = segmentation[y, x]
                seg_dx = segmentation[y, x + segmentation_radius] if x + segmentation_radius < w else segmentation[y, x]
                seg_dy = segmentation[y + segmentation_radius, x] if y + segmentation_radius < h else segmentation[y, x]
                seg_dx_1 = segmentation[y, x - segmentation_radius] if x - segmentation_radius >= 0 else segmentation[y, x]
                seg_dy_1 = segmentation[y - segmentation_radius, x] if y - segmentation_radius >= 0 else segmentation[y, x]
                if seg != segmentation_class_id or \
                    seg_dx != segmentation_class_id or seg_dy != segmentation_class_id or \
                    seg_dx_1 != segmentation_class_id or seg_dy_1 != segmentation_class_id:
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
    return np.degrees(angles)

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

def plot_histogram_with_image(normal_image, normal_angles, title="Normals Histogram"):
    valid_angles = normal_angles[np.isfinite(normal_angles)]
    if valid_angles.size == 0:
        print("No valid angles found for histogram.")
        return
    
    angles_degrees = valid_angles
    
    # Subplots, one for normal_image and one for histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(normal_image)
    ax1.set_title("Normal Image")
    ax1.axis('off')
    ax2.set_title(title)
    ax2.hist(angles_degrees.flatten(), bins=50, color='blue', alpha=0.7)
    ax2.set_xlabel("Normal Values")
    ax2.set_ylabel("Frequency")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(HISTOGRAM_OUTPUT_PATH, f"{title.replace(' ', '_')}.png"))

if __name__=="__main__":
    # Example usage
    print("Surface normal estimation script executed successfully.")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset file {DATASET_PATH} does not exist.")
        exit(-1)
    if not os.path.exists(DATASET_CSV_PATH):
        print(f"Dataset CSV file {DATASET_CSV_PATH} does not exist.")
        exit(-1)
    if not os.path.exists(HISTOGRAM_OUTPUT_PATH):
        os.makedirs(HISTOGRAM_OUTPUT_PATH)
    dataset = pd.read_csv(DATASET_CSV_PATH)
    dataset = dataset.sort_values(by='location_timestamp').reset_index(drop=True)
    dataset = dataset[DATASET_COLS]
    
    fx, fy = 1335.0, 1335.0
    cx, cy = 960.0, 720.0
    segmentation_class_id = 22 # sidewalk

    step = 4
    scale = 4

    for data_row in tqdm(dataset.itertuples(), desc="Processing dataset rows", total=len(dataset)):
        rgb_path = getattr(data_row, 'rgb_frame_path')
        depth_path = getattr(data_row, 'depth_frame_path')
        segmentation_path = getattr(data_row, 'annotation_frame_path')
        
        paths = [rgb_path, depth_path, segmentation_path]
        
        rgb_path = os.path.join(DATASET_PATH, paths[0].lstrip('/'))
        depth_path = os.path.join(DATASET_PATH, paths[1].lstrip('/'))
        segmentation_path = os.path.join(DATASET_PATH, paths[2].lstrip('/'))
        
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path) or not os.path.exists(segmentation_path):
            print(f"Missing files for row {data_row.Index}: {rgb_path}, {depth_path}, {segmentation_path}")
            continue
        
        rgb = Image.open(rgb_path)
        depth = Image.open(depth_path)
        rgb = rgb.resize(depth.size, Image.BILINEAR)
        # depth = depth.resize(rgb.size, Image.NEAREST)
        segmentation = Image.open(segmentation_path)
        segmentation = segmentation.resize(depth.size, Image.NEAREST)

        rgb = np.array(rgb)
        depth = np.array(depth)
        segmentation = np.array(segmentation)
    
        segmentation = filter_segmentation(segmentation, segmentation_class_id, (0, 0.5, 1, 0.9))
        normals = compute_surface_normals(depth, fx, fy, cx, cy, step=step, 
                                        segmentation=segmentation, segmentation_radius=5)
        normal_angles = get_normal_angles(normals)
        # print_normals_statistics(normal_angles)
        normal_image = visualize_normals_on_image(rgb, normals, step=step, scale=scale)
        plot_histogram_with_image(normal_image, normal_angles, 
                                  title=f"{data_row.Index}: {data_row.rgb_frame_path.split('/')[-1]}")