import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os

def get_segmentation_mask(segmentation, class_ids: list, bounds: tuple):
    min_x, min_y, max_x, max_y = bounds
    min_x = min_x * segmentation.shape[1]
    min_y = min_y * segmentation.shape[0]
    max_x = max_x * segmentation.shape[1]
    max_y = max_y * segmentation.shape[0]
    
    print(f"Filtering segmentation for class {class_ids} within bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")

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

def get_visualize_image(rgb_path, segmentation_path, segmentation_class_ids: list, bounds: tuple) -> Image.Image:
    """
    Visualizes the RGB image with an overlay of the specified segmentation class.
    Segmentation has a lower alpha to allow the RGB image to show through.
    """
    rgb = Image.open(rgb_path).convert('RGBA')  # Ensure the image has an alpha channel
    segmentation = Image.open(segmentation_path)
    
    # Convert segmentation to numpy array for processing
    segmentation = np.array(segmentation)
    
    # Filter the segmentation for the specified class
    filtered_segmentation_mask = get_segmentation_mask(segmentation, segmentation_class_ids, bounds)
    
    # Create an overlay image with lower alpha
    overlay = np.zeros_like(rgb, dtype=np.float32)
    overlay[filtered_segmentation_mask] = [0, 0, 255, 128]  # Blue color with alpha

    overlay_image = Image.fromarray(overlay.astype(np.uint8), 'RGBA')
    combined_image = Image.alpha_composite(rgb, overlay_image)

    return combined_image

DATASET_PATH = 'dataset/'
DATASET_CSV_PATH = 'dataset/dataset.csv'

DATASET_COLS = [
    'rgb_frame_path',
    'depth_frame_path',
    'annotation_frame_path',
    'odometry_timestamp',
    'location_timestamp'
]
SIDEWALK_SURFACE_INTEGRITY_COL = 'sidewalk_surface_integrity'

SURFACE_INTEGRITY_OPTIONS = ['Broken', 'Gap', 'Occluded', 'Correct', 'Not Sure']

dataset = pd.read_csv(DATASET_CSV_PATH)
if SIDEWALK_SURFACE_INTEGRITY_COL not in dataset.columns:
    dataset[SIDEWALK_SURFACE_INTEGRITY_COL] = None
dataset = dataset.sort_values(by='location_timestamp').reset_index(drop=True)

# Find the next unlabeled image
unlabeled = dataset[dataset[SIDEWALK_SURFACE_INTEGRITY_COL].isnull()]
print(f"Found {len(unlabeled)} unlabeled images.")

index = -1
st.title("Image Labeling: Sidewalk Integrity")
if unlabeled.empty:
    st.success("All images are labeled.")
    st.stop()
else:
    rgb_path = None
    while rgb_path is None and index < len(dataset) - 1:
        index += 1
        idx = dataset.index[index]
        sidewalk_status = dataset.loc[idx, SIDEWALK_SURFACE_INTEGRITY_COL]
        # Check if sidewalk status is not labeled (nan)
        print(sidewalk_status)
        if not(pd.isna(sidewalk_status) or sidewalk_status == '' or sidewalk_status == 'None'):
            continue
        rgb_path = os.path.join(DATASET_PATH, dataset.loc[idx, 'rgb_frame_path'].lstrip('/'))
        if not os.path.exists(rgb_path):
            rgb_path = None
            
    if rgb_path is None:
        st.error("No more unlabeled images found.")
        st.stop()
        
st.write(f"Labeling image {index + 1} of {len(dataset)}")
# image = Image.open(rgb_path)
segmentation_path = os.path.join(DATASET_PATH, dataset.loc[idx, 'annotation_frame_path'].lstrip('/'))
segmentation_class_id = [22, 9, 25]  # sidewalk, curb ramp, tactile paving
bounds = (0, 0.5, 1, 0.9)
image = get_visualize_image(
    rgb_path,
    segmentation_path,
    segmentation_class_id,
    bounds
)
st.image(image, caption=f"Image {index + 1}", width=300)

cols = st.columns(len(SURFACE_INTEGRITY_OPTIONS))
for i, option in enumerate(SURFACE_INTEGRITY_OPTIONS):
    with cols[i]:
        if st.button(option, key=f"{index}_{option}"):
            dataset.loc[index, SIDEWALK_SURFACE_INTEGRITY_COL] = option
            dataset.to_csv(DATASET_CSV_PATH, index=False)
            st.success(f"Marked as {option}")
            st.rerun()