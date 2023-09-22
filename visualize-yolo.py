# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import glob
import json
import os
import shutil
from PIL import Image
import numpy as np
import torch
import torch.distributed
import torch.utils.data
import yaml
from tqdm import tqdm
from yolov4_pytorch.utils import yoloxywh2xyxy
from yolov4_pytorch.utils import plot_one_box

from pathlib import Path
import cv2
import random


def visualize_yolo_bbox(image_path, label_path=None, names=None):

    raw_image = cv2.imread(image_path)
    print(raw_image.shape, Image.open(image_path).size)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    with open(label_path, 'r') as f:
        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels

    print(l)

    for row in l:
        class_id = int(row[0])
        bbox = row[1:]
        h, w = raw_image.shape[0:2]

        xyxy = yoloxywh2xyxy(np.array([bbox]), w, h)[0]
        print(h, w, xyxy)
        plot_one_box(xyxy=xyxy,
                        image=raw_image,
                        color=colors[class_id],
                        label=names[class_id],
                        line_thickness=3)

    cv2.imwrite(f'outputs/{Path(image_path).name}', raw_image)

if __name__ == "__main__":
    pid = '41HvNhtQ9zL'
    visualize_yolo_bbox(f'data/logo-729k/images/{pid}.jpg', f'data/logo-729k/labels/{pid}.txt', ['logo'])
