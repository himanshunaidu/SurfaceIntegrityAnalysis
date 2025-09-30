"""
This script analyzes a 3D mesh to check for surface integrity issues.
"""
import os
import glob
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN