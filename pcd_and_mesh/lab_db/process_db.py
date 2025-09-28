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

from utils.read_utils import read_pcd, read_rgb, read_mesh, read_transform, read_intrinsics, read_main_data

"""
NOTE: The point cloud is read from a dedicated point cloud directory.
Other data (RGB, mesh) is read from the main trial directory.
"""

APRIL_TAGS_USED = {
    "BL": 201,
    "BR": 202,
    "M": 102, # Middle
    "TL": 203,
    "TR": 204
}

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

def detect_apriltags(img: Image.Image, aruco_detector=None) -> tuple[np.ndarray, np.ndarray]:
    if aruco_detector is None:
        dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(dic, parameters)

    img_array = np.array(img)
    image_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    
    return corners, ids

if __name__=="__main__":
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled", "experiment_4"))
    ROW_LABEL = "0-3-3-0-b"
    
    MAIN_PATH = os.path.join(DATASET_PATH, "main")
    PCD_PATH = os.path.join(DATASET_PATH, "pcd")
    
    img, mesh, transform, intrinsics = read_main_data(MAIN_PATH, ROW_LABEL)
    # mesh.compute_vertex_normals()
    pcd = read_pcd(PCD_PATH, ROW_LABEL)
    # pcd, adjusted_transform = align_pcd(pcd, transform)
    
    frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, np.zeros(3))
    reference_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.2, np.zeros(3))

    # o3d.visualization.draw_geometries([pcd, mesh, frame_mesh.transform(adjusted_transform), reference_frame_mesh])

    april_corners, april_ids = detect_apriltags(img)
    
    if april_ids is None: exit("No AprilTags detected.")
    april_centroids = [np.mean(corner[0], axis=0) for corner in april_corners]
    
    # Mark the centroids in the mesh using intrinsics and transform
    april_centroids_3d = []
    april_meshes = []
    for i, corner in enumerate(april_corners):
        tag_id = str(april_ids[i][0])
        
        centroid_2d = np.mean(corner[0], axis=0)
        x_2d, y_2d = int(centroid_2d[0]), int(centroid_2d[1])
        
        depth = 1.0  # Assume a nominal depth of 1 meter
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_3d = (x_2d - cx) * depth / fx
        y_3d = (y_2d - cy) * depth / fy
        z_3d = depth
        
        point_camera = np.array([x_3d, y_3d, z_3d, 1.0])
        point_world = transform @ point_camera
        
        tag_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        tag_mesh.paint_uniform_color([1, 0, 0])
        tag_mesh.translate(point_world[:3])
        april_meshes.append(tag_mesh)
        print(f"AprilTag ID {tag_id} at 2D pixel ({x_2d}, {y_2d}) corresponds to 3D point {point_world[:3]}")
        
    o3d.visualization.draw_geometries([mesh, *april_meshes, frame_mesh.transform(transform), reference_frame_mesh])