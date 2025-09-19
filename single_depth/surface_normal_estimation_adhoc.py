import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

# paths = ['adhoc/rgb/3f76035400_frame_001440.png', 
#          'adhoc/depth/3f76035400_depth_frame_001440.png',
#          'adhoc/segment/3f76035400_frame_001440_gtFine_labelIds.png']
# paths = ['adhoc/rgb/2f6aa18afe_frame_003240.png', 
#          'adhoc/depth/2f6aa18afe_depth_frame_003240.png',
#          'adhoc/segment/2f6aa18afe_frame_003240_gtFine_labelIds.png']
# paths = ['adhoc/rgb/6024554a53_frame_014400.png', 
#          'adhoc/depth/6024554a53_depth_frame_014400.png',
#          'adhoc/segment/6024554a53_frame_014400_gtFine_labelIds.png']
# paths = ['adhoc/rgb/3aaf8437bf_frame_004320.png', 
#          'adhoc/depth/3aaf8437bf_depth_frame_004320.png',
#          'adhoc/segment/3aaf8437bf_frame_004320_gtFine_labelIds.png']
paths = ['adhoc/rgb/50d8803b77_frame_000000.png',
         'adhoc/depth/50d8803b77_depth_frame_000000.png',
         None]

rgb_path = paths[0]
depth_path = paths[1]
rgb = Image.open(rgb_path)
depth = Image.open(depth_path)
rgb = rgb.resize(depth.size, Image.BILINEAR)
# depth = depth.resize(rgb.size, Image.NEAREST)

segmentation_path = paths[2] if paths[2] is not None else None
# segmentation_path = 'adhoc/segment/2f6aa18afe_frame_003240_gtFine_labelIds.png'
# segmentation_path = 'adhoc/segment/6024554a53_frame_014400_gtFine_labelIds.png'
# segmentation_path = 'adhoc/segment/3aaf8437bf_frame_004320_gtFine_labelIds.png'
segmentation = None
if segmentation_path is not None:
    segmentation = Image.open(segmentation_path)
    segmentation = segmentation.resize(depth.size, Image.NEAREST)

rgb = np.array(rgb)
depth = np.array(depth)
if segmentation_path is not None:
    segmentation = np.array(segmentation)

fx, fy = 1335.0, 1335.0
cx, cy = 960.0, 720.0
segmentation_class_id = 22 # sidewalk

print(f"Image size: {rgb.shape}, Depth size: {depth.shape}, Segmentation size: {segmentation.shape if segmentation is not None else 'None'}")

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

def filter_segmentation_when_none(rgb, class_id, bounds: tuple):
    min_x, min_y, max_x, max_y = bounds
    min_x = min_x * rgb.shape[1]
    min_y = min_y * rgb.shape[0]
    max_x = max_x * rgb.shape[1]
    max_y = max_y * rgb.shape[0]

    print(f"Filtering segmentation for class {class_id} within bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")

    mask = (np.arange(rgb.shape[0])[:, None] >= min_y) & \
        (np.arange(rgb.shape[0])[:, None] < max_y) & \
        (np.arange(rgb.shape[1])[None, :] >= min_x) & \
        (np.arange(rgb.shape[1])[None, :] < max_x)
    filtered = np.ones((rgb.shape[0], rgb.shape[1], 1)) * class_id
    filtered[~mask] = 0
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

def visualize_normals_on_image(rgb, normals, step=20, scale=20, normal_angles=None):
    percentiles = np.percentile(normal_angles, [0, 25, 50, 75, 100]) if normal_angles is not None else None
    colors = [(0, 255, 255), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (165, 42, 42)]
    up = np.array([0, 0, 1])
    
    vis = rgb.copy()
    for y in range(step, normals.shape[0] - step, step):
        for x in range(step, normals.shape[1] - step, step):
            n = normals[y, x]
            angle = np.degrees(np.arccos(np.clip(np.dot(n, up), -1.0, 1.0))) if np.linalg.norm(n) > 0 else None
            color = colors[np.digitize(angle, percentiles) - 1] if percentiles is not None and angle is not None else (255, 255, 255)
            if np.linalg.norm(n) > 0:
                end_point = (int(x + scale * n[0]), int(y - scale * n[1]))
                cv2.arrowedLine(vis, (x, y), end_point, color=color, thickness=scale//4, tipLength=scale/10)    
    return vis

def plot_histogram_with_image(normal_image, normal_angles, title="Normals Histogram"):
    valid_angles = normal_angles[np.isfinite(normal_angles)]
    if valid_angles.size == 0:
        print("No valid angles found for histogram.")
        return
    
    angles_degrees = valid_angles
    
    # Subplots, one for normal_image and one for histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Normal Image")
    ax1.axis('off')
    ax2.set_title(title)
    ax2.hist(angles_degrees.flatten(), bins=50, color='blue', alpha=0.7)
    ax2.set_xlabel("Normal Values")
    ax2.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

step = 4
scale = 4
# exit(-1)

if segmentation_path is not None:
    segmentation = filter_segmentation(segmentation, segmentation_class_id, (0, 0.3, 1, 0.9))
else:
    segmentation = filter_segmentation_when_none(rgb, segmentation_class_id, (0, 0.5, 1, 1))
normals = compute_surface_normals(depth, fx, fy, cx, cy, step=step, 
                                  segmentation=segmentation, segmentation_radius=5)
normal_angles = get_normal_angles(normals)
print_normals_statistics(normal_angles)
normal_image = visualize_normals_on_image(rgb, normals, step=step, scale=scale,
                                          normal_angles=normal_angles)
plot_histogram_with_image(normal_image, normal_angles, title="Surface Normals Histogram")

# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB))
# plt.title("Surface Normals Visualized on RGB Image")
# plt.axis('off')
# plt.show()

# Comments
# rgb_path = 'adhoc/rgb/3f76035400_frame_001440.png': breakages: [0.05573232 0.03906335 0.03994829]
# rgb_path = 'adhoc/rgb/2f6aa18afe_frame_003240.png': gap: [0.13096857 0.1425487  0.05064309]
# rgb_path = 'adhoc/rgb/6024554a53_frame_014400.png': breakages: [0.04355213 0.00648725 0.02388454]
# rgb_path = 'adhoc/rgb/3aaf8437bf_frame_004320.png': correct: [0.03739867 0.0079297  0.01975509]