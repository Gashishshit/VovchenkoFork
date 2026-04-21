import os
import sys
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_device, load_config
from src.dataset import get_dataloaders
from src.model import DETROreDetector


def denormalize_box(box_norm, img_w, img_h):
    cx, cy, w, h = box_norm
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_boxes(image, boxes, labels, scores, color=(0, 255, 0)):
    h, w = image.shape[:2]

    for box, label, score in zip(boxes, labels, scores):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().item()
        if isinstance(score, torch.Tensor):
            score = score.cpu().item()

        x1, y1, x2, y2 = denormalize_box(box, w, h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"Class {label}: {score:.2f}"
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    config = load_config(args.config_path)
    device = get_device()

    os.makedirs(args.output_dir, exist_ok=True)

    _, val_loader = get_dataloaders(config)

    model = DETROreDetector(
        num_classes=config["model"]["num_classes"], model_name=config["model"]["name"]
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"✓ Model loaded from {args.model_path}")

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets, names) in enumerate(val_loader):
            images = images.to(device)
            predictions = model.predict(images, threshold=args.threshold)

            for idx, (image, pred, target, name) in enumerate(
                zip(images, predictions, targets, names)
            ):
                image_np = image.cpu().numpy().transpose(1, 2, 0)
                image_np = (image_np * 255).astype(np.uint8)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                image_with_pred = draw_boxes(
                    image_np.copy(),
                    pred["boxes"],
                    pred["labels"],
                    pred["scores"],
                    color=(0, 255, 0),
                )

                image_with_target = draw_boxes(
                    image_np.copy(),
                    target["boxes"],
                    target["labels"],
                    np.ones(len(target["labels"])),
                    color=(255, 0, 0),
                )

                cv2.imwrite(
                    f"{args.output_dir}/pred_{batch_idx}_{idx}_{name}", image_with_pred
                )
                cv2.imwrite(
                    f"{args.output_dir}/target_{batch_idx}_{idx}_{name}",
                    image_with_target,
                )

            if batch_idx >= 5:
                break

    print(f"✓ Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
