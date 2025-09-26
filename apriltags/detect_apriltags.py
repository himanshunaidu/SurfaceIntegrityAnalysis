import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

IMG_PATH = "img/capture_1.JPG"

def ensure_apriltag_dict():
    # OpenCV ≥4.7 includes AprilTag dictionaries in aruco
    if not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        raise RuntimeError(
            "Your OpenCV build lacks AprilTag dictionaries. "
            "Install/upgrade with: pip install --upgrade opencv-contrib-python"
        )
        
def detect_apriltags(image_path: str):
    ensure_apriltag_dict()
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dic, parameters)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None:
        print(f"Detected {len(ids)} AprilTag(s): {ids.flatten().tolist()}")
        print("Corners:", corners, "\nIDs:", ids)
        img_marked = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
    else:
        print("No AprilTags detected.")
        img_marked = img

    # Show the image with detected markers
    cv2.imshow("Detected AprilTags", img_marked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    detect_apriltags(IMG_PATH)