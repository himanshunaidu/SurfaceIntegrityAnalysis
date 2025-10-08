"""
This script contains utility functions for cropping meshes and point clouds based on specified anchors.
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
import open3d as o3d

def get_directions(camera_transform: np.ndarray) -> dict:
    """
    Get the direction vectors (right, forward) from the camera transformation matrix.
    These directions should be parallel to the ground plane. (Y axis is up)
    """
    if camera_transform is None or camera_transform.shape != (4, 4):
        raise ValueError("Invalid camera transformation matrix.")
    
    # The right direction is the first column of the rotation part of the transform
    right = camera_transform[0:3, 0]
    right[1] = 0  # Project onto the ground plane (Y=0)
    right /= np.linalg.norm(right)
    # The forward direction is the negative of the third column of the rotation part of the transform
    ## The camera transform's Z axis points backwards, so we take the negative
    forward = -camera_transform[0:3, 2]
    forward[1] = 0  # Project onto the ground plane (Y=0)
    forward /= np.linalg.norm(forward)
    
    return {"right": right, "forward": forward}


def _get_line_equation(point1: np.ndarray, point2: np.ndarray) -> tuple[float, float, float]:
    """
    Get the line equation coefficients (A, C, D) for the line passing through point1 and point2.
    The line equation is given by: Ax + Cz + D = 0
    (The line is represented in the XZ plane, ignoring the Y coordinate)
    """
    if point1 is None or point2 is None:
        return None
    if point1.shape != (3,) or point2.shape != (3,):
        return None
    
    # Direction vector from point1 to point2
    direction = point2 - point1
    if np.linalg.norm(direction) == 0:
        raise ValueError("Input points must be distinct.")
    
    # Normal vector to the line in the XZ plane
    A = -direction[2]  # -dz
    C = direction[0]   # dx
    D = -(A * point1[0] + C * point1[2])

    return float(A), float(C), float(D)

def _estimate_line_from_parallel_line(point1: np.ndarray, point2: np.ndarray, parallel_line: tuple[float, float, float], distance: float) -> tuple[float, float, float]:
    """
    Estimate a line equation that is parallel to the given line and at a specified distance from it.
    
    Args:
        point1 (np.ndarray): A point on the new line (3D coordinates).
        point2 (np.ndarray): Another point on the new line (3D coordinates).
        parallel_line (tuple): Coefficients (A, C, D) of the line to be paralleled.
        distance (float): The distance from the parallel line.
        
    Returns:
        tuple: Coefficients (A, C, D) of the estimated line.
    """
    if (point1 is None and point2 is None) or parallel_line is None or distance is None:
        return None
    if len(parallel_line) != 3:
        return None
    
    point = point1 if (point1 is not None and point1.shape == (3,)) else point2
    if point is None or point.shape != (3,): return None
    
    A, C, D = parallel_line
    
    # Find the equation that is parallel to the given line and passes through point
    D_new = -(A * point[0] + C * point[2])
    return float(A), float(C), float(D_new)

def _estimate_line_from_direction(point1: np.ndarray, point2: np.ndarray, direction: np.ndarray, distance: float) -> tuple[float, float, float]:
    """
    Estimate a line equation that is in the direction of the given vector and at a specified distance from the point.
    
    Args:
        point1 (np.ndarray): A point on the new line (3D coordinates). (left or bottom point)
        point2 (np.ndarray): Another point on the new line (3D coordinates). (right or top point)
        direction (np.ndarray): Direction vector (should be in XZ plane).
        distance (float): The distance from the point to the line.
        
    Returns:
        tuple: Coefficients (A, C, D) of the estimated line.
    """
    if (point1 is None and point2 is None) or direction is None or distance is None:
        return None
    if direction.shape != (3,):
        return None
    
    point = point1 if (point1 is not None and point1.shape == (3,)) else point2
    if point is None or point.shape != (3,): return None
    # If point1 is None, and point2 is used as the point, then we need to reverse the direction
    if point1 is None:
        direction = -direction

    # Project direction onto XZ plane
    direction[1] = 0
    if np.linalg.norm(direction) == 0:
        return None
    direction /= np.linalg.norm(direction)
    
    # Normal vector to the line in the XZ plane
    A = -direction[2]  # -dz
    C = direction[0]   # dx
    D = -(A * point[0] + C * point[2])
    
    return float(A), float(C), float(D)

def _verify_line_equation(line: tuple[float, float, float], point: np.ndarray, tolerance: float = 1e-3):
    """
    Verify if a point lies on the line defined by the given line equation coefficients (A, C, D).
    
    Args:
        line (tuple): Coefficients (A, C, D) of the line equation Ax + Cz + D = 0.
        point (np.ndarray): A point in 3D space (x, y, z).
        tolerance (float): Acceptable tolerance for floating-point comparison.

    Returns:
        bool: True if the point lies on the line, False otherwise.
    """
    if line is None or point is None or point.shape != (3,):
        return False

    A, C, D = line
    x, y, z = point

    # Check if the point satisfies the line equation within the given tolerance
    verification = abs(A * x + C * z + D) < tolerance
    print(f"Verifying point {point} on line {line}: {'Pass' if verification else 'Fail'}")

def _line_intersection(line1: tuple[float, float, float], line2: tuple[float, float, float]) -> np.ndarray:
    """
    Calculate the intersection point of two lines defined by their equations.
    
    Args:
        line1 (tuple): Coefficients (A1, C1, D1) of the first line equation A1*x + C1*z + D1 = 0.
        line2 (tuple): Coefficients (A2, C2, D2) of the second line equation A2*x + C2*z + D2 = 0.
        
    Returns:
        np.ndarray: The intersection point (x, y, z) in 3D space. Y is set to 0.
    """
    if line1 is None or line2 is None:
        return None
    if len(line1) != 3 or len(line2) != 3:
        return None
    
    A1, C1, D1 = line1
    A2, C2, D2 = line2
    
    # Calculate the determinant
    determinant = A1 * C2 - A2 * C1
    if abs(determinant) < 1e-6:
        print("Lines are parallel or coincident; no unique intersection.")
        return None  # Lines are parallel or coincident
    
    # Calculate intersection coordinates
    x = (C1 * D2 - C2 * D1) / determinant
    z = (A2 * D1 - A1 * D2) / determinant
    
    return np.array([x, 0.0, z])  # Y is set to 0

def get_crop_anchors(top_left: np.ndarray = None, top_right: np.ndarray = None,
                     bottom_left: np.ndarray = None, bottom_right: np.ndarray = None,
                     horizontal_length: float = 0.6, vertical_length: float = 1.2,
                     camera_transform: np.ndarray = None) -> tuple:
    """
    Get the crop anchors from the given corner points.
    
    This function calculates the crop anchors based on the provided corner points.
    If not all the corner points are provided, it estimates the missing ones. 
        The function assumes that all the points have around the same y-coordinate (which is the vertical axis).
        The horizontal_length and vertical_length parameters define the expected distances between the anchors.
    The anchors are returned as a dictionary with keys 'top_left', 'top_right', 'bottom_left', 'bottom_right'.
    
    Estimation logic:
    - If a corner point is missing, it is estimated based on the available points and the defined lengths.
    - For example, if the top_left point is missing, it can be estimated as:
      top_left = top_right - horizontal_length
      - Similarly, other missing points can be estimated.
    (If even one of the points are given, along with potential lengths and camera_transform,
    the function can estimate the rest of the points with a degree of error.)
    """
    # Check if any of the corner points are provided
    if not any(pt is not None for pt in [top_left, top_right, bottom_left, bottom_right]):
        return None
    
    # Estimate the distances
    horizontal_length_1 = np.linalg.norm(top_right - top_left) if top_left is not None and top_right is not None else None
    horizontal_length_2 = np.linalg.norm(bottom_right - bottom_left) if bottom_left is not None and bottom_right is not None else None
    horizontal_length_est = np.mean([l for l in [horizontal_length_1, horizontal_length_2] if l is not None])
    if horizontal_length_est is not None:
        horizontal_length = horizontal_length_est
    
    vertical_length_1 = np.linalg.norm(top_left - bottom_left) if top_left is not None and bottom_left is not None else None
    vertical_length_2 = np.linalg.norm(top_right - bottom_right) if top_right is not None and bottom_right is not None else None
    vertical_length_est = np.mean([l for l in [vertical_length_1, vertical_length_2] if l is not None])
    if vertical_length_est is not None:
        vertical_length = vertical_length_est
    
    # Estimate the side line equations
    horizontal_top_line = _get_line_equation(top_left, top_right)
    horizontal_bottom_line = _get_line_equation(bottom_left, bottom_right)
    vertical_left_line = _get_line_equation(top_left, bottom_left)
    vertical_right_line = _get_line_equation(top_right, bottom_right)
    
    # Estimate the missing side line equations using parallel lines and distances
    if horizontal_top_line is None:
        horizontal_top_line = _estimate_line_from_parallel_line(top_left, top_right, horizontal_bottom_line, horizontal_length)
    if horizontal_bottom_line is None:
        horizontal_bottom_line = _estimate_line_from_parallel_line(bottom_left, bottom_right, horizontal_top_line, horizontal_length)
    if vertical_left_line is None:
        vertical_left_line = _estimate_line_from_parallel_line(top_left, bottom_left, vertical_right_line, vertical_length)
    if vertical_right_line is None:
        vertical_right_line = _estimate_line_from_parallel_line(top_right, bottom_right, vertical_left_line, vertical_length)
        
    # Estimate the side line equations that are still missing, using the lengths and directions from camera_transform
    try:
        directions = get_directions(camera_transform) if camera_transform is not None else None
    except Exception as e:
        print("Error in getting directions from camera transform:", e)
        return None
    if horizontal_top_line is None:
        horizontal_top_line = _estimate_line_from_direction(top_left, top_right, directions["right"], horizontal_length) if directions is not None else None
    if horizontal_bottom_line is None:
        horizontal_bottom_line = _estimate_line_from_direction(bottom_left, bottom_right, directions["right"], horizontal_length) if directions is not None else None
    if vertical_left_line is None:
        vertical_left_line = _estimate_line_from_direction(top_left, bottom_left, directions["forward"], vertical_length) if directions is not None else None
    if vertical_right_line is None:
        vertical_right_line = _estimate_line_from_direction(top_right, bottom_right, directions["forward"], vertical_length) if directions is not None else None

    print("Horizontal length:", horizontal_length, "Vertical length:", vertical_length)
    
    print(f"Horizontal Top Line: {horizontal_top_line}, Top Left: {top_left}, Top Right: {top_right}")
    print(f"Horizontal Bottom Line: {horizontal_bottom_line}, Bottom Left: {bottom_left}, Bottom Right: {bottom_right}")
    print(f"Vertical Left Line: {vertical_left_line}, Top Left: {top_left}, Bottom Left: {bottom_left}")
    print(f"Vertical Right Line: {vertical_right_line}, Top Right: {top_right}, Bottom Right: {bottom_right}")
    
    # Checking tolerance
    print("\nChecking line equations with tolerance...")
    tolerance = 1e-2
    _verify_line_equation(horizontal_top_line, top_left, tolerance)
    _verify_line_equation(horizontal_top_line, top_right, tolerance)
    _verify_line_equation(horizontal_bottom_line, bottom_left, tolerance)
    _verify_line_equation(horizontal_bottom_line, bottom_right, tolerance)
    _verify_line_equation(vertical_left_line, top_left, tolerance)
    _verify_line_equation(vertical_left_line, bottom_left, tolerance)
    _verify_line_equation(vertical_right_line, top_right, tolerance)
    _verify_line_equation(vertical_right_line, bottom_right, tolerance)
    
    # Calculate the intersection points of the lines to get the corner points
    if top_left is None:
        top_left = _line_intersection(horizontal_top_line, vertical_left_line)
    if top_right is None:
        top_right = _line_intersection(horizontal_top_line, vertical_right_line)
    if bottom_left is None:
        bottom_left = _line_intersection(horizontal_bottom_line, vertical_left_line)
    if bottom_right is None:
        bottom_right = _line_intersection(horizontal_bottom_line, vertical_right_line)

    top_left[1] = top_right[1] = bottom_left[1] = bottom_right[1] = np.mean([pt[1] for pt in [top_left, top_right, bottom_left, bottom_right] if pt is not None and pt[1] != 0.0])

    return (top_left, top_right, bottom_left, bottom_right)