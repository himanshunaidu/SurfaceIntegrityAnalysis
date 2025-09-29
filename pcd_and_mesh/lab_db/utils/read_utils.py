import os
import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
import open3d as o3d

TRANSFORM_COLUMNS_TO_MATRIX_MAP = {
    "rxx": (0, 0),
    "rxy": (1, 0),
    "rxz": (2, 0),
    "ryx": (0, 1),
    "ryy": (1, 1),
    "ryz": (2, 1),
    "rzx": (0, 2),
    "rzy": (1, 2),
    "rzz": (2, 2),
    "x": (0, 3),
    "y": (1, 3),
    "z": (2, 3)
}

def read_pcd(dir_path: str, row_trial_name: str) -> o3d.geometry.PointCloud:
    """
    Read a point cloud file (PCD format) and return an Open3D PointCloud object.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"PCD dir not found: {dir_path}")

    pcd = o3d.io.read_point_cloud(os.path.join(dir_path, f"{row_trial_name}.ply"))
    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {row_trial_name}")

    print(f"Loaded point cloud from {os.path.join(dir_path, f'{row_trial_name}.ply')} with {len(pcd.points)} points for trial '{row_trial_name}'")
    return pcd

def read_rgb(dir_path: str, row_trial_name: str) -> Image.Image:
    """
    Reads the RGB image from the given directory.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Main dir not found: {dir_path}")

    # Currently, the rgb image can have any name, but it is placed alone in a folder as a png image.
    img_files = glob.glob(os.path.join(dir_path, "*.png"))
    if not img_files:
        raise FileNotFoundError(f"No PNG image found in dir: {dir_path}")
    img_path = img_files[0]

    img = Image.open(img_path)
    print(f"Loaded RGB image from {img_path} with size {img.size} for trial '{row_trial_name}'")
    return img

def read_depth(dir_path: str, row_trial_name: str) -> np.ndarray:
    """
    Reads the depth image from the given directory.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Main dir not found: {dir_path}")

    # Currently, the depth image can have any name, but it is placed alone in a folder as a png image.
    depth_files = glob.glob(os.path.join(dir_path, "*.png"))
    if not depth_files:
        raise FileNotFoundError(f"No PNG depth image found in dir: {dir_path}")
    depth_path = depth_files[0]

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to load depth image: {depth_path}")

    print(f"Loaded depth image from {depth_path} with shape {depth.shape} for trial '{row_trial_name}'")
    print(f"Image format: {depth.dtype}, min depth: {np.min(depth)}, max depth: {np.max(depth)}")
    return depth

def read_confidence(dir_path: str, row_trial_name: str) -> np.ndarray:
    """
    Reads the confidence image from the given directory.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Main dir not found: {dir_path}")

    # Currently, the confidence image can have any name, but it is placed alone in a folder as a png image.
    conf_files = glob.glob(os.path.join(dir_path, "*.png"))
    if not conf_files:
        raise FileNotFoundError(f"No PNG confidence image found in dir: {dir_path}")
    conf_path = conf_files[0]

    confidence = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)
    if confidence is None:
        raise ValueError(f"Failed to load confidence image: {conf_path}")

    print(f"Loaded confidence image from {conf_path} with shape {confidence.shape} for trial '{row_trial_name}'")
    print(f"Image format: {confidence.dtype}, min confidence: {np.min(confidence)}, max confidence: {np.max(confidence)}")
    return confidence

def read_mesh(dir_path: str, row_trial_name: str) -> o3d.geometry.TriangleMesh:
    """
    Reads the mesh file (PLY format) from the given directory.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Main dir not found: {dir_path}")

    # Currently, the mesh file can have any name, but it is placed alone in a folder as a ply file.
    mesh_files = glob.glob(os.path.join(dir_path, "*.ply"))
    if not mesh_files:
        raise FileNotFoundError(f"No PLY mesh file found in dir: {dir_path}")
    mesh_path = mesh_files[0]

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"Loaded mesh is empty: {mesh_path}")

    print(f"Loaded mesh from {mesh_path} with {len(mesh.triangles)} triangles for trial '{row_trial_name}'")
    return mesh

def read_transform(dir_path: str, row_trial_name: str) -> np.ndarray:
    """
    Reads the camera-to-world transformation matrix from a text file.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Main dir not found: {dir_path}")

    transform_path = os.path.join(dir_path, f"camera_transform.csv")
    if not os.path.exists(transform_path):
        raise FileNotFoundError(f"Transform file not found: {transform_path}")
    
    transform_df = pd.read_csv(transform_path, delimiter=', ')
    # Will later convert to required format.
    transform_row = transform_df.iloc[0].to_dict()
    transform = np.eye(4)
    for col, (i, j) in TRANSFORM_COLUMNS_TO_MATRIX_MAP.items():
        if col not in transform_row:
            raise ValueError(f"Column '{col}' not found in transform file: {transform_path}")
        transform[i, j] = transform_row[col]
        
    # Invert the y-axis and z-axis to convert from ARKit to Open3D coordinate system.
    # convert_matrix = np.diag([1, -1, -1, 1])
    # transform = convert_matrix @ transform @ convert_matrix

    print(f"Loaded transform matrix from {transform_path} for trial '{row_trial_name}':\n{transform}")
    return transform

def read_intrinsics(dir_path: str, row_trial_name: str) -> np.ndarray:
    """
    Reads the camera intrinsics matrix from a text file.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Main dir not found: {dir_path}")

    intrinsics_path = os.path.join(dir_path, f"camera_matrix.csv")
    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
    
    # intrinsics = pd.read_csv(intrinsics_path, header=None)
    intrinsics = np.loadtxt(intrinsics_path, delimiter=',')

    print(f"Loaded intrinsics matrix from {intrinsics_path} for trial '{row_trial_name}':\n{intrinsics}")
    return intrinsics

def read_main_data(dir_path: str, row_trial_name: str) -> tuple[Image.Image, np.ndarray, o3d.geometry.TriangleMesh, pd.DataFrame, pd.DataFrame]:
    """
    Reads the RGB image and mesh file (PLY format) from the main trial directory.
    """
    row_path = os.path.join(dir_path, row_trial_name)
    if not os.path.exists(row_path):
        raise FileNotFoundError(f"Trial dir not found: {row_path}")
    
    img_dir_path = os.path.join(row_path, "rgb")
    mesh_dir_path = os.path.join(row_path, "mesh")
    depth_dir_path = os.path.join(row_path, "depth")
    confidence_dir_path = os.path.join(row_path, "confidence")

    img = read_rgb(img_dir_path, row_trial_name)
    mesh = read_mesh(mesh_dir_path, row_trial_name)
    transform = read_transform(row_path, row_trial_name)
    intrinsics = read_intrinsics(row_path, row_trial_name)
    depth = read_depth(depth_dir_path, row_trial_name)
    # confidence = read_confidence(confidence_dir_path, row_trial_name)

    return img, depth, mesh, transform, intrinsics