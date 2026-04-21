import os
import sys
import torch
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_device, load_config
from src.model import DETROreDetector


def inference_on_image(model, image_path, device, threshold=0.5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose(
        [
            A.Resize(height=800, width=800),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    predictions = model.predict(image_tensor, threshold=threshold)
    return predictions[0], image


def denormalize_box(box_norm, img_w, img_h):
    cx, cy, w, h = box_norm
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def visualize_predictions(image, predictions, threshold=0.5):
    class_names = {0: "Арматура", 1: "Большие камни"}
    colors = {0: "red", 1: "orange"}

    h, w = image.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    for box, label, score in zip(
        predictions["boxes"], predictions["labels"], predictions["scores"]
    ):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().item()
        if isinstance(score, torch.Tensor):
            score = score.cpu().item()

        x1, y1, x2, y2 = denormalize_box(box, w, h)

        rect_width = x2 - x1
        rect_height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            rect_width,
            rect_height,
            linewidth=2,
            edgecolor=colors[label],
            facecolor="none",
        )
        ax.add_patch(rect)

        class_name = class_names.get(label, f"Class {label}")
        text = f"{class_name}: {score:.2f}"
        ax.text(
            x1,
            y1 - 10,
            text,
            fontsize=10,
            color=colors[label],
            bbox=dict(facecolor="white", alpha=0.7),
        )

    ax.set_title(f"Обнаружено объектов: {len(predictions['scores'])}", fontsize=14)
    ax.axis("off")

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/first_model.pth")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config_path)
    device = get_device()

    model = DETROreDetector(
        num_classes=config["model"]["num_classes"], model_name=config["model"]["name"]
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k[6:]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    model.model.load_state_dict(new_state_dict, strict=False)
    print(f"✓ Model loaded from {args.model_path}")

    predictions, image = inference_on_image(
        model, args.image_path, device, args.threshold
    )

    print(f"\n=== Predictions ===")
    print(f"Found {len(predictions['scores'])} objects")
    for i, (score, label, box) in enumerate(
        zip(predictions["scores"], predictions["labels"], predictions["boxes"])
    ):
        print(
            f"Object {i}: Class {label.item()}, Score {score.item():.4f}, Box {box.tolist()}"
        )

    fig = visualize_predictions(image, predictions, args.threshold)

    if args.save_path:
        fig.savefig(args.save_path, bbox_inches="tight", dpi=100)
        print(f"✓ Visualization saved to {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
