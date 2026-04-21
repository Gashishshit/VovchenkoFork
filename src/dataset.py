import os
import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from src.augmentations import get_train_transforms, get_val_transforms


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None, img_size=800):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.img_size = img_size
        self.image_files = sorted([f.name for f in Path(images_dir).glob("*.jpg")])
        print(f"Found {len(self.image_files)} images in {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        label_name = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(self.labels_dir, label_name)

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])
                        x1 = max(0.0, (x_c - w / 2) * orig_w)
                        y1 = max(0.0, (y_c - h / 2) * orig_h)
                        x2 = min(orig_w, (x_c + w / 2) * orig_w)
                        y2 = min(orig_h, (y_c + h / 2) * orig_h)
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls)

        boxes = (
            np.array(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 4), dtype=np.float32)
        )
        labels = (
            np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64)
        )

        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=boxes.tolist(), class_labels=labels.tolist()
            )
            image = transformed["image"]
            boxes = (
                np.array(transformed["bboxes"], dtype=np.float32)
                if transformed["bboxes"]
                else np.zeros((0, 4), dtype=np.float32)
            )
            labels = (
                np.array(transformed["class_labels"], dtype=np.int64)
                if transformed["class_labels"]
                else np.zeros(0, dtype=np.int64)
            )

        _, h, w = image.shape
        boxes_detr = np.zeros_like(boxes)
        if len(boxes) > 0:
            boxes_detr[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
            boxes_detr[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
            boxes_detr[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w
            boxes_detr[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h
            boxes_detr = np.clip(boxes_detr, 0.0, 1.0)

        target = {
            "boxes": torch.from_numpy(boxes_detr).float(),
            "labels": torch.from_numpy(labels).long(),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "size": torch.tensor([h, w]),
        }

        return image, target, img_name


def collate_fn(batch):
    images, targets, names = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets), list(names)


def get_dataloaders(config):
    train_dataset = YOLODataset(
        f"{config['data']['dataset_dir']}/images/train",
        f"{config['data']['dataset_dir']}/labels/train",
        transforms=get_train_transforms(config["data"]["img_size"]),
        img_size=config["data"]["img_size"],
    )

    val_dataset = YOLODataset(
        f"{config['data']['dataset_dir']}/images/val",
        f"{config['data']['dataset_dir']}/labels/val",
        transforms=get_val_transforms(config["data"]["img_size"]),
        img_size=config["data"]["img_size"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
