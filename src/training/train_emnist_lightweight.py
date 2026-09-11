from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data.emnist_dataset import get_emnist_dataloaders
from src.models.emnist_lightweight_cnn import EMNISTLightweightCNN
from src.training.trainer import train_model
from src.utils.seed import set_seed
from src.utils.parameter_counter import check_parameter_limit


def main():
    config_path = Path("configs/emnist_lightweight_cnn.yaml")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    set_seed(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("EMNIST Lightweight CNN Training")
    print("=" * 60)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = get_emnist_dataloaders(
        data_dir="data/raw",
        batch_size=config["batch_size"],
        validation_size=10000,
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    model = EMNISTLightweightCNN(
        num_classes=config["num_classes"]
    ).to(device)

    parameter_report = check_parameter_limit(
        model,
        config["max_trainable_parameters"]
    )

    print(
        f"Trainable parameters: "
        f"{parameter_report['trainable_parameters']:,}"
    )

    print(
        f"Maximum allowed: "
        f"{parameter_report['max_parameters']:,}"
    )

    if not parameter_report["passed"]:
        raise ValueError(
            "Lightweight CNN exceeds the 100,000 parameter limit."
        )

    print("Parameter constraint: PASSED")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Epochs: {config['epochs']}")

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["epochs"],
        model_path=config["model_path"],
    )

    print("\nTraining completed.")


if __name__ == "__main__":
    main()