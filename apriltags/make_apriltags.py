import argparse
import math
import re
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

DPI = 300

# In pixels
def get_image_size() -> Tuple[int, int]:
    # A4 size in mm
    width_mm = 210
    height_mm = 297
    return (mm_to_pixels(width_mm), mm_to_pixels(height_mm))

def mm_to_pixels(mm: float) -> int:
    return int(mm / 25.4 * DPI)

def ensure_apriltag_dict():
    # OpenCV ≥4.7 includes AprilTag dictionaries in aruco
    if not hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
        raise RuntimeError(
            "Your OpenCV build lacks AprilTag dictionaries. "
            "Install/upgrade with: pip install --upgrade opencv-contrib-python"
        )

def draw_apriltag(tag_id: int, side_px: int) -> Image.Image:
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = np.zeros((side_px, side_px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dic, int(tag_id), side_px, img, 1)
    return Image.fromarray(img)

def layout_tags(
    page_px: Tuple[int, int],
    margin_px: int,
    tag_px: int,
    spacing_px: int,
    ids: List[int],
    labels: List[str],
    font_path: str = None
) -> Image.Image:
    W, H = page_px
    canvas = Image.new("L", (W, H), color=255)  # white
    draw = ImageDraw.Draw(canvas)
    
    # Render each tag
    tags = [draw_apriltag(tid, tag_px) for tid in ids]
    
    placements = {
        "TL": (margin_px, margin_px),
        "TR": (W - margin_px - tag_px, margin_px),
        "BR": (W - margin_px - tag_px, H - margin_px - tag_px),
        "BL": (margin_px, H - margin_px - tag_px),
    }
    
    label_to_img = dict(zip(labels, tags))
    
    for lab, (x, y) in placements.items():
        canvas.paste(label_to_img[lab], (x, y))
        
    # Add guides
    guide_len = int(tag_px * 0.25)
    guide_w = max(1, tag_px // 60)
    guides = [
        # top-left
        ((margin_px - spacing_px, margin_px - spacing_px),
         (margin_px - spacing_px + guide_len, margin_px - spacing_px)),
        ((margin_px - spacing_px, margin_px - spacing_px),
         (margin_px - spacing_px, margin_px - spacing_px + guide_len)),

        # top-right
        ((W - margin_px + spacing_px, margin_px - spacing_px),
         (W - margin_px + spacing_px - guide_len, margin_px - spacing_px)),
        ((W - margin_px + spacing_px, margin_px - spacing_px),
         (W - margin_px + spacing_px, margin_px - spacing_px + guide_len)),

        # bottom-right
        ((W - margin_px + spacing_px, H - margin_px + spacing_px),
         (W - margin_px + spacing_px - guide_len, H - margin_px + spacing_px)),
        ((W - margin_px + spacing_px, H - margin_px + spacing_px),
         (W - margin_px + spacing_px, H - margin_px + spacing_px - guide_len)),

        # bottom-left
        ((margin_px - spacing_px, H - margin_px + spacing_px),
         (margin_px - spacing_px + guide_len, H - margin_px + spacing_px)),
        ((margin_px - spacing_px, H - margin_px + spacing_px),
         (margin_px - spacing_px, H - margin_px + spacing_px - guide_len)),
    ]
    for p0, p1 in guides:
        draw.line([p0, p1], fill=0, width=guide_w)
    
    try:
        fnt = ImageFont.truetype(font_path or "DejaVuSans.ttf", size=max(14, tag_px // 10))
    except Exception:
        fnt = ImageFont.load_default()

    label_offset = int(tag_px * 0.05)
    text_fill = 0
    for lab, (x, y) in placements.items():
        tid = ids[labels.index(lab)]
        text = f"{lab}  (ID {tid})"
        tw = draw.textlength(text, font=fnt)
        th = fnt.size
        draw.rectangle([x, y + tag_px + label_offset, x + tw + 8, y + tag_px + label_offset + th + 4], fill=255)
        draw.text((x + 4, y + tag_px + label_offset + 2), text, fill=text_fill, font=fnt)
    
    return canvas

if __name__ == "__main__":
    ensure_apriltag_dict()
    
    size = get_image_size()
    
    tag_size_mm = 70
    ids = [301, 302, 303, 304]
    labels = ["TL", "TR", "BR", "BL"]
    margin_mm = 25
    spacing_mm = 15
    
    fontpath = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    
    canvas = layout_tags(
        page_px=size,
        margin_px=mm_to_pixels(margin_mm),
        tag_px=mm_to_pixels(tag_size_mm),
        spacing_px=mm_to_pixels(spacing_mm),
        ids=ids,
        labels=labels,
        font_path=fontpath
    )
    canvas.save("apriltags/apriltag_sheet_3.png")
    canvas.show()