from pathlib import Path

import torch
from tqdm import tqdm

from src.models.model_utils import save_checkpoint, save_model_info


def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    device,
    model_path,
    max_trainable_parameters=None,
):
    if max_trainable_parameters is not None:
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        if trainable >= max_trainable_parameters:
            raise ValueError(
                f"Lightweight model has {trainable:,} trainable parameters. "
                f"Required: < {max_trainable_parameters:,}."
            )

    model.to(device)

    best_val_accuracy = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )

        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f} | "
            f"val acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                model,
                model_path,
                {
                    "epoch": epoch,
                    "best_val_accuracy": best_val_accuracy,
                },
            )

    save_model_info(
        Path(model_path).with_name("model_info.json"),
        model,
        {"best_validation_accuracy": best_val_accuracy},
    )

    return history
