# Point Cloud and Mesh Analysis

Analyses point clouds and polygon meshes for surface integrity of sidewalks.
Currently, only contains ad-hoc scripts to visualize and manipulate point clouds and polygon meshes.

## Lab-Controlled Datasets

_This dataset is WIP_

This submodule will also help create and maintain lab-controlled datasets for surface integrity analysis of sidewalks.

The lab experiments will consist of using some surface boards to simulate different sidewalk surface conditions (e.g., cracks, bumps, slopes, etc.) and capturing the data using iOS devices.

### Factors

- Issue Characteristics: Gap Width, Gap Depth, Gap Orientation, Gap Length, Surface Height Difference
- Device Characteristics: Device Height, Device Angle, Device Speed

(Future)
- Additional Issue Characteristics: Gap Slope
- Environment Characteristics: Lighting Conditions, Weather Conditions (may not be relevant for lab-controlled datasets)

### Dataset Contents

Each row in the dataset would consist of the following:
- RGB Image
- Depth Image
- Depth Confidence Image
- Camera Intrinsics (from ARKit)
- Camera Transform (from ARKit)
- Device Location (Latitude, Longitude, Altitude)
- Issue Characteristics: Gap Width, Gap Depth, Gap Orientation, Gap Length, Surface Height Difference (and others)
- Device Characteristics: Device Height, Device Angle, Device Speed (and others)

### Dataset Format

The dataset will currently be stored in a folder structure as follows:

```
lab_controlled_datasets/
    ├── experiment_x/
    │   ├── rgb/
    │   │   ├── frame_0001.png
    │   │   ├── frame_0002.png
    │   │   └── ...
    │   ├── depth/
    │   │   ├── frame_0001.png
    │   │   ├── frame_0002.png
    │   │   └── ...
    │   ├── confidence/
    │   │   ├── frame_0001.png
    │   │   ├── frame_0002.png
    │   │   └── ...
    │   ├── dataset.csv
    │   └── ...
    └── ...
```

The `dataset.csv` file will be the main source of data, containing all the metadata and paths to the images.
The CSV file will have the following columns:
- frame_id: Unique identifier for each frame (e.g., frame_0001)
- rgb_path: Path to the RGB image
- depth_path: Path to the Depth image
- confidence_path: Path to the Depth Confidence image
- camera_intrinsics: Camera intrinsics matrix (as a fixed list of columns)
- camera_transform: Camera transform matrix (as a fixed list of columns)
- device_location: Device location (Latitude, Longitude, Altitude)
- issue_characteristics: Gap Width, Gap Depth, Gap Orientation, Gap Length, Surface Height Difference (and others)
- device_characteristics: Device Height, Device Angle, Device Speed (and others)
- surface_integrity_label: Label for the surface integrity (e.g., good, minor_issue, major_issue)

The images will be stored in their respective folders, and the paths in the CSV file will be relative to the `experiment_x` folder.