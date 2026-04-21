import os
import sys
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device, load_config
from src.dataset import get_dataloaders
from src.model import DETROreDetector


def train_epoch(model, train_loader, optimizer, device, grad_clip=0.1):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc="Training")
    for images, targets, _ in pbar:
        images = images.to(device)

        targets_detr = []
        for t in targets:
            targets_detr.append(
                {"boxes": t["boxes"].to(device), "class_labels": t["labels"].to(device)}
            )

        outputs = model(images, labels=targets_detr)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


def main():
    config = load_config("config.yaml")
    set_seed(config["data"]["seed"])
    device = get_device()

    os.makedirs(config["paths"]["checkpoints_dir"], exist_ok=True)

    train_loader, val_loader = get_dataloaders(config)

    model = DETROreDetector(
        num_classes=config["model"]["num_classes"], model_name=config["model"]["name"]
    ).to(device)

    param_dicts = model.get_optimizer_params()
    optimizer = AdamW(
        param_dicts,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = StepLR(
        optimizer,
        step_size=config["training"]["scheduler_step"],
        gamma=config["training"]["scheduler_gamma"],
    )

    best_loss = float("inf")

    for epoch in range(config["training"]["num_epochs"]):
        avg_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=config["training"]["grad_clip"],
        )
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config["paths"]["model_path"])
            print(f"✓ Best model saved (loss={best_loss:.4f})")

        scheduler.step()


if __name__ == "__main__":
    main()
