#!/usr/bin/env python3
import random
import shutil
from pathlib import Path

SOURCE = Path("/home/pony/School/Final_Project/data/herbaldata/train")
TARGET = Path("/home/pony/School/Final_Project/data/herbaldata_split")
TRAIN_RATIO = 0.8
random.seed(42)

for cls_dir in SOURCE.iterdir():
    if not cls_dir.is_dir():
        continue
    images = sorted([p for p in cls_dir.iterdir() if p.is_file()])
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    splits = {
        TARGET / "train" / cls_dir.name: images[:split_idx],
        TARGET / "val" / cls_dir.name: images[split_idx:]
    }
    for dst_dir, files in splits.items():
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, dst_dir / img.name)