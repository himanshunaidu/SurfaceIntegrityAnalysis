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
    
    return A, C, D
    

def get_crop_anchors(top_left: np.ndarray = None, top_right: np.ndarray = None,
                     bottom_left: np.ndarray = None, bottom_right: np.ndarray = None,
                     horizontal_length: float = 0.6, vertical_length: float = 1.2) -> dict:
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
    """
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
    
    return None