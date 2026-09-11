from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data.emnist_dataset import get_emnist_dataloaders
from src.models.emnist_standard_cnn import EMNISTStandardCNN
from src.training.trainer import train_model
from src.utils.seed import set_seed


def main():

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    config_path = Path(
        "configs/emnist_standard_cnn.yaml"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    set_seed(
        config["seed"]
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("EMNIST Improved Standard CNN Training")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader
    ) = get_emnist_dataloaders(
        data_dir="data/raw",
        batch_size=config["batch_size"],
        validation_size=10000,
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = EMNISTStandardCNN(
        num_classes=config["num_classes"]
    ).to(device)

    # ---------------------------------------------------------
    # Loss function
    # ---------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # ---------------------------------------------------------
    # Learning-rate scheduler
    # ---------------------------------------------------------

    scheduler_config = config["scheduler"]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
        min_lr=scheduler_config["min_lr"],
    )

    # ---------------------------------------------------------
    # Model information
    # ---------------------------------------------------------

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    print(
        f"Maximum epochs: "
        f"{config['epochs']}"
    )

    print(
        f"Initial learning rate: "
        f"{config['learning_rate']}"
    )

    print(
        f"Scheduler patience: "
        f"{scheduler_config['patience']}"
    )

    print(
        f"Early stopping patience: "
        f"{config['early_stopping_patience']}"
    )

    print(
        f"Model path: "
        f"{config['model_path']}"
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"],
        model_path=config["model_path"],
        scheduler=scheduler,
        early_stopping_patience=
            config["early_stopping_patience"],
    )

    # ---------------------------------------------------------
    # Finished
    # ---------------------------------------------------------

    print(
        "\nImproved Standard CNN training completed."
    )


if __name__ == "__main__":

    main()