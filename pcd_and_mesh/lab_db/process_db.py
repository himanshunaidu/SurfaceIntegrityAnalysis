"""
This script reads input data captured using an iPhone device.
The data includes RGB and mesh data from SurfaceIntegrityChecker iOS app, and point cloud data from apps such as SiteScape.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
import open3d as o3d

"""
NOTE: The point cloud is read from a dedicated point cloud directory.
Other data (RGB, mesh) is read from the main trial directory.
"""
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

def align_pcd(pcd: o3d.geometry.PointCloud, transform: np.ndarray) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Align the point cloud to the camera coordinate system using the given transformation matrix.
    """
    if pcd.is_empty():
        raise ValueError("Input point cloud is empty.")
    
    if transform.shape != (4, 4):
        raise ValueError(f"Transform matrix must be 4x4, got shape {transform.shape}")
    
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location()
    
    # Remove translation component for alignment
    adjusted_transform = transform.copy()
    adjusted_transform[:3, 3] = 0
    print(f"Initial transform (no translation):\n{adjusted_transform}")

    # Invert the y-axis and z-axis to convert from ARKit to Open3D coordinate system.
    # convert_matrix = np.diag([1, -1, -1, 1])
    # adjusted_transform = convert_matrix @ transform @ convert_matrix
    
    # The ARKit camera transform was calculated in portrait mode, so we need to rotate the point cloud accordingly.
    # This rotation is 90 degrees around the z-axis, with x changing to -y and y changing to x.
    rotation_90_z = np.array(
        [[0, 1, 0, 0],
         [-1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    )
    adjusted_transform = adjusted_transform @ np.linalg.inv(rotation_90_z)
    print(f"Applying adjusted transform to point cloud:\n{adjusted_transform}")
    
    Y = np.array([0, 1, 0])
    
    # Orient the transform's x-axis to point parallel to the ground plane.
    # Thus, its x and z components remain the same, but y component is adjusted to be 0.
    # Then, orient the transform's y-axis to point upwards (opposite to gravity).
    # Thus, rotate the transform around x-axis by pitch = - current_pitch
    rotation = adjusted_transform[:3, :3]
    x, y, z = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    x_new = np.array([x[0], 0., x[2]])
    x_new /= np.linalg.norm(x_new)
    z_new = np.cross(x_new, Y)
    z_new /= np.linalg.norm(z_new)
    y_new = np.cross(z_new, x_new)
    y_new /= np.linalg.norm(y_new)
    
    adjusted_transform[:3, 0] = x_new
    adjusted_transform[:3, 1] = y_new
    adjusted_transform[:3, 2] = z_new
    print(f"Final transform after aligning x-axis to ground plane and y-axis to gravity:\n{adjusted_transform}")

    pcd_homogeneous = np.hstack((np.asarray(pcd.points), np.ones((len(pcd.points), 1))))
    aligned_points_homogeneous = (adjusted_transform @ pcd_homogeneous.T).T
    aligned_points = aligned_points_homogeneous[:, :3]

    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)
    if pcd.has_colors():
        aligned_pcd.colors = pcd.colors
    if pcd.has_normals():
        aligned_pcd.normals = pcd.normals

    return aligned_pcd, adjusted_transform

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

def read_main_data(dir_path: str, row_trial_name: str) -> tuple[Image.Image, o3d.geometry.TriangleMesh, pd.DataFrame, pd.DataFrame]:
    """
    Reads the RGB image and mesh file (PLY format) from the main trial directory.
    """
    row_path = os.path.join(dir_path, row_trial_name)
    if not os.path.exists(row_path):
        raise FileNotFoundError(f"Trial dir not found: {row_path}")
    
    img_dir_path = os.path.join(row_path, "rgb")
    mesh_dir_path = os.path.join(row_path, "mesh")

    img = read_rgb(img_dir_path, row_trial_name)
    mesh = read_mesh(mesh_dir_path, row_trial_name)
    transform = read_transform(row_path, row_trial_name)
    intrinsics = read_intrinsics(row_path, row_trial_name)

    return img, mesh, transform, intrinsics

if __name__=="__main__":
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled", "experiment_4"))
    ROW_LABEL = "0-0-0-0-b"
    
    MAIN_PATH = os.path.join(DATASET_PATH, "main")
    PCD_PATH = os.path.join(DATASET_PATH, "pcd")
    
    img, mesh, transform, intrinsics = read_main_data(MAIN_PATH, ROW_LABEL)
    pcd = read_pcd(PCD_PATH, ROW_LABEL)
    pcd, adjusted_transform = align_pcd(pcd, transform)
    
    frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, np.zeros(3))
    reference_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.2, np.zeros(3))

    o3d.visualization.draw_geometries([pcd, mesh, frame_mesh.transform(adjusted_transform), reference_frame_mesh])