import torch
import glob
import os
import json
import math
from argparse import ArgumentParser
import time
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from tqdm import tqdm

import coremltools as ct
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

IMG_PATH = "../dataset/broken_sidewalk_2/rgb/50d8803b77_frame_000000.png"
IMG_PATH = "../dataset/ios_point_mapper/leftImg8bit/default/1e8fde45bd_frame_006840_leftImg8bit.png"
# IMG_PATH = "../dataset/China_MotorBike_001977.jpg"
MODEL_NAME = "v8n_175_16_960.mlpackage"

model = ct.models.MLModel(MODEL_NAME)
img = Image.open(IMG_PATH).convert("RGB")

img_size = img.size
lower_side = min(img_size)
# Crop at center to make it square
left = (img_size[0] - lower_side) / 2
top = (img_size[1] - lower_side) / 2
right = (img_size[0] + lower_side) / 2
bottom = (img_size[1] + lower_side) / 2
img = img.crop((left, top, right, bottom))

img = img.resize((640, 640))
# img = img.resize((640, 640))
# Crop in center of img
# img = F.to_tensor(img).unsqueeze(0).numpy()
pred = model.predict({"image": img, "iouThreshold": 0.45, "confidenceThreshold": 0.25})

print(pred["coordinates"].shape)
print(pred["confidence"].shape)

# print(pred["coordinates"])

# Draw boxes
draw = ImageDraw.Draw(img)
index = 0
for box, score in zip(pred["coordinates"], pred["confidence"]):
    x, y, w, h = box
    print(f"Box {index}: ", x, y, w, h)
    x, y, w, h = x * 640, y * 640, w * 640, h * 640
    x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
    print(f"Box corrected bounds {index}: ", x1, y1, w, h, "\n")
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    # draw.text((x1, y1), f"{score:.2f}", fill="red")
    draw.text((x1, y1), f"Index: {index}", fill="red")
    # print(score)
    index += 1
img.show()