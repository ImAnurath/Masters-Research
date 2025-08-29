import os
import json
from tqdm import tqdm
from PIL import Image

# Root path
root = os.path.dirname(__file__)

# Dataset root
dataset_root = os.path.join(root, "data_split")  # contains train/, val/, test/

# Path to YOLO classes.txt
classes_txt = os.path.join(dataset_root, "classes.txt")

# Read class names 
with open(classes_txt) as f:
    class_names = [line.strip() for line in f.readlines()]

categories = [
    {"id": i, "name": name, "supercategory": name}
    for i, name in enumerate(class_names)
]

def yolo_to_coco(split):
    """Convert one split (train/val/test) into COCO format"""
    images_dir = os.path.join(dataset_root, split, "images")
    labels_dir = os.path.join(dataset_root, split, "labels")

    coco = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1

    for idx, img_name in enumerate(tqdm(os.listdir(images_dir), desc=f"Processing {split}")):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

        # Get image size
        with Image.open(img_path) as im:
            w, h = im.size

        coco["images"].append({
            "id": idx,
            "file_name": img_name,
            "width": w,
            "height": h
        })

        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.split())
                cls = int(cls)

                # YOLO (cx, cy, w, h relative) → COCO (x,y,w,h absolute)
                x_min = (x - bw / 2) * w
                y_min = (y - bh / 2) * h
                box_w = bw * w
                box_h = bh * h

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": cls,
                    "bbox": [x_min, y_min, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0
                })
                ann_id += 1

    # Save JSON
    out_path = os.path.join(dataset_root, f"{split}.json")
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"Saved {split}.json with {len(coco['images'])} images and {len(coco['annotations'])} annotations")


# Run for train/val/test
for split in ["train", "val", "test"]:
    yolo_to_coco(split)
