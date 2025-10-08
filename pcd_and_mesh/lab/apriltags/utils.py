import numpy as np
from PIL import Image
import cv2

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