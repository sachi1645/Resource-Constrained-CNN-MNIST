from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.data.dataset import get_dataloaders
from src.models.lightweight_cnn import LightweightCNN
from src.models.model_utils import count_parameters
from src.training.trainer import train_model
from src.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[2]


def main():
    config = yaml.safe_load(
        (ROOT / "configs" / "lightweight_cnn.yaml").read_text()
    )

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LightweightCNN()
    trainable = count_parameters(model)

    print(f"Lightweight trainable parameters: {trainable:,}")

    if trainable >= config["max_trainable_parameters"]:
        raise ValueError(
            f"Parameter constraint failed: {trainable:,} >= "
            f"{config['max_trainable_parameters']:,}"
        )

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config["epochs"],
        device=device,
        model_path=ROOT / config["model_path"],
        max_trainable_parameters=config["max_trainable_parameters"],
    )

    print(f"Training complete. Best model: {config['model_path']}")
    return history


if __name__ == "__main__":
    main()
