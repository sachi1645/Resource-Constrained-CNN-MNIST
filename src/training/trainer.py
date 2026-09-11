from pathlib import Path

import torch
from tqdm import tqdm

from src.models.model_utils import save_checkpoint, save_model_info


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training=True,
):
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for images, labels in tqdm(
            loader,
            leave=False
        ):

            images = images.to(device)
            labels = labels.to(device)

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if training:

                loss.backward()

                optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


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
    scheduler=None,
    early_stopping_patience=None,
):

    # ---------------------------------------------------------
    # Parameter constraint
    # ---------------------------------------------------------

    if max_trainable_parameters is not None:

        trainable = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        if trainable >= max_trainable_parameters:

            raise ValueError(
                f"Model has {trainable:,} "
                f"trainable parameters. "
                f"Required: < "
                f"{max_trainable_parameters:,}."
            )

    model.to(device)

    # ---------------------------------------------------------
    # Training state
    # ---------------------------------------------------------

    best_val_accuracy = 0.0

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    history = []

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1
    ):

        # -----------------------------------------------------
        # Training
        # -----------------------------------------------------

        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
        )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        val_loss, val_accuracy = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
        )

        # -----------------------------------------------------
        # Save history
        # -----------------------------------------------------

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        history.append(row)

        # -----------------------------------------------------
        # Current learning rate
        # -----------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f} | "
            f"val acc={val_accuracy:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # -----------------------------------------------------
        # Best model
        # -----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_val_loss = val_loss

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                model_path,
                {
                    "epoch": epoch,
                    "best_val_accuracy":
                        best_val_accuracy,
                    "best_val_loss":
                        best_val_loss,
                },
            )

            print(
                f"  ✓ New best model "
                f"(val acc: "
                f"{best_val_accuracy * 100:.2f}%)"
            )

        else:

            epochs_without_improvement += 1

        # -----------------------------------------------------
        # Learning-rate scheduler
        # -----------------------------------------------------

        if scheduler is not None:

            scheduler.step(val_loss)

            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != current_lr:

                print(
                    f"  ↓ Learning rate reduced: "
                    f"{current_lr:.6f} → "
                    f"{new_lr:.6f}"
                )

        # -----------------------------------------------------
        # Early stopping
        # -----------------------------------------------------

        if (
            early_stopping_patience is not None
            and epochs_without_improvement
            >= early_stopping_patience
        ):

            print(
                f"\nEarly stopping triggered "
                f"after {epoch} epochs."
            )

            print(
                f"Best validation accuracy: "
                f"{best_val_accuracy * 100:.2f}%"
            )

            break

    # ---------------------------------------------------------
    # Save model information
    # ---------------------------------------------------------

    save_model_info(
        Path(model_path).with_name(
            "model_info.json"
        ),
        model,
        {
            "best_validation_accuracy":
                best_val_accuracy,
            "best_validation_loss":
                best_val_loss,
        },
    )

    return history