import os
import sys
import torch
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_device, load_config
from src.dataset import get_dataloaders
from src.model import DETROreDetector
from src.metrics import compute_all_metrics


def evaluate(model, val_loader, device, threshold=0.5):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            predictions = model.predict(images, threshold=threshold)

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    metrics = compute_all_metrics(all_predictions, all_targets, threshold)
    return metrics, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.config_path)
    device = get_device()

    _, val_loader = get_dataloaders(config)

    model = DETROreDetector(
        num_classes=config["model"]["num_classes"], model_name=config["model"]["name"]
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"✓ Model loaded from {args.model_path}")

    metrics, _, _ = evaluate(model, val_loader, device, args.threshold)

    print("\n=== Evaluation Results ===")
    print(f"FPR:       {metrics['fpr']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(
        f"\nTP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}"
    )


if __name__ == "__main__":
    main()
