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

TEST_APRIL_TAGS_USED = {
    "TL": 301,
    "TR": 302,
    "BL": 304,
    "BR": 303
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

def get_depth_at_img_point(depth: np.ndarray, img: Image.Image, x: int, y: int) -> float:
    """
    Get the depth value at the given image pixel coordinates (x, y).
    The depth image size is not the same as the RGB image size. 
    It is a fraction of the RGB image size, so we need to scale the coordinates accordingly.
    """
    depth_height, depth_width = depth.shape
    img_width, img_height = img.size
    print(f"Depth image size: ({depth_width}, {depth_height}), RGB image size: ({img_width}, {img_height})")

    x_depth = int(x * depth_width / img_width)
    y_depth = int(y * depth_height / img_height)

    if x_depth < 0 or x_depth >= depth_width or y_depth < 0 or y_depth >= depth_height:
        raise ValueError(f"Depth coordinates out of bounds: ({x_depth}, {y_depth}) for depth size ({depth_width}, {depth_height})")

    depth_value = depth[y_depth, x_depth]
    return float(depth_value)

def get_point_2d_from_pixel(img: Image.Image, x: int, y: int) -> np.ndarray:
    """
    Get the 2D point from the given image pixel coordinates (x, y).
    
    Currently, the image is not resized or rotated, so the pixel coordinates are the same as the image coordinates.
    """
    img_width, img_height = img.size

    if x < 0 or x >= img_width or y < 0 or y >= img_height:
        raise ValueError(f"Image coordinates out of bounds: ({x}, {y}) for image size ({img_width}, {img_height})")
    
    return np.array([x, y])

def draw_circle_on_image(img: Image.Image, x: int, y: int, radius: int = 10, color: tuple = (255, 0, 0)) -> Image.Image:
    """
    Draw a circle on the image at the given pixel coordinates (x, y).
    """
    img_with_circle = img.copy()
    draw = ImageDraw.Draw(img_with_circle)
    left_up_point = (x - radius, y - radius)
    right_down_point = (x + radius, y + radius)
    draw.ellipse([left_up_point, right_down_point], outline=color, width=3)
    return img_with_circle

def get_world_point_from_depth(depth: np.ndarray, img: Image.Image, intrinsics: np.ndarray, transform: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Get the world point from the given depth image and camera intrinsics at the pixel coordinates (x, y).
    """
    depth_value = get_depth_at_img_point(depth, img, x, y) / 1000.0  # Convert mm to meters
    image_point = np.array([x, y, 1.0])
    ray = np.linalg.inv(intrinsics) @ image_point
    ray /= np.linalg.norm(ray)
    
    camera_point = ray * depth_value
    camera_point[1] *= -1  # Invert y-axis to convert from ARKit to Open3D coordinate system.
    camera_point[2] *= -1  # Invert z-axis to convert from ARKit to Open3D coordinate system.
    
    camera_point_homogeneous = np.hstack((camera_point, 1.0))
    
    world_point = transform @ camera_point_homogeneous
    return world_point[:3]
    

def main(dataset_path: str, row_label: str, april_tags_used: dict = APRIL_TAGS_USED):
    main_path = os.path.join(dataset_path, "main")
    pcd_path = os.path.join(dataset_path, "pcd")
    april_tags_used_positions = {v: k for k, v in april_tags_used.items()}

    img, depth, mesh, transform, intrinsics = read_main_data(main_path, row_label)
    # mesh.compute_vertex_normals()
    pcd = read_pcd(pcd_path, row_label)
    # pcd, adjusted_transform = align_pcd(pcd, transform)
    inverted_intrinsics = np.linalg.inv(intrinsics)
    
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
        tag_position = april_tags_used_positions.get(int(tag_id), None)
        if tag_position is None:
            print(f"Warning: Detected AprilTag ID {tag_id} not in the list of used tags. Skipping this tag.")
            continue
        
        centroid_2d = np.mean(corner[0], axis=0)
        x_2d, y_2d = int(centroid_2d[0]), int(centroid_2d[1])
        
        x_2d, y_2d = get_point_2d_from_pixel(img, x_2d, y_2d)
        point_world = get_world_point_from_depth(depth, img, intrinsics, transform, x_2d, y_2d)
        april_centroids_3d.append(point_world)

        tag_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        if tag_position == "BL": tag_mesh.paint_uniform_color([0, 1, 0])
        elif tag_position == "BR": tag_mesh.paint_uniform_color([0, 0, 1])
        else: tag_mesh.paint_uniform_color([1, 0, 0])
        tag_mesh.translate(point_world[:3])
        april_meshes.append(tag_mesh)
        print(f"AprilTag ID {tag_id} at 2D pixel ({x_2d}, {y_2d}) corresponds to 3D point {point_world[:3]}")
        
    o3d.visualization.draw_geometries([mesh, *april_meshes, frame_mesh.transform(transform), reference_frame_mesh])

if __name__=="__main__":
    # DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "lab_controlled", "experiment_4"))
    # ROW_LABEL = "0-3-3-0-b"
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "test"))
    ROW_LABEL = "Test"

    main(DATASET_PATH, ROW_LABEL, april_tags_used=TEST_APRIL_TAGS_USED)