from pathlib import Path

import torch
import torch.nn as nn
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix

from src.data.emnist_dataset import get_emnist_dataloaders
from src.models.emnist_standard_cnn import EMNISTStandardCNN
from src.models.emnist_lightweight_cnn import EMNISTLightweightCNN
from src.models.model_utils import count_parameters, model_size_mb
from src.utils.seed import set_seed


def evaluate_model(model, test_loader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                labels.cpu().numpy()
            )

    test_loss = total_loss / total
    test_accuracy = correct / total

    cm = confusion_matrix(
        all_targets,
        all_predictions
    )

    return test_loss, test_accuracy, cm


def load_model(model_class, model_path, num_classes, device):
    model = model_class(
        num_classes=num_classes
    ).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    return model


def main():

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("EMNIST MODEL EVALUATION")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # Load EMNIST test dataset
    _, _, test_loader = get_emnist_dataloaders(
        data_dir="data/raw",
        batch_size=128,
        validation_size=10000,
        num_workers=0,
        seed=42,
    )

    models = {
        "Standard CNN": (
            EMNISTStandardCNN,
            "models/emnist_standard_cnn/best_model.pth"
        ),
        "Lightweight CNN": (
            EMNISTLightweightCNN,
            "models/emnist_lightweight_cnn/best_model.pth"
        ),
    }

    results = {}

    for model_name, (model_class, model_path) in models.items():

        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        model = load_model(
            model_class,
            model_path,
            num_classes=47,
            device=device,
        )

        parameters = count_parameters(model)
        size_mb = model_size_mb(model)

        test_loss, test_accuracy, cm = evaluate_model(
            model,
            test_loader,
            device,
        )

        results[model_name] = {
            "parameters": parameters,
            "model_size_mb": size_mb,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "confusion_matrix": cm,
        }

        print(f"Parameters : {parameters:,}")
        print(f"Model size : {size_mb:.4f} MB")
        print(f"Test loss  : {test_loss:.4f}")
        print(
            f"Test accuracy : {test_accuracy * 100:.2f}%"
        )

        print("\nConfusion Matrix:")
        print(cm)

    # Comparison
    standard = results["Standard CNN"]
    lightweight = results["Lightweight CNN"]

    parameter_reduction = (
        standard["parameters"] /
        lightweight["parameters"]
    )

    size_reduction = (
        standard["model_size_mb"] /
        lightweight["model_size_mb"]
    )

    accuracy_gap = (
        standard["test_accuracy"] -
        lightweight["test_accuracy"]
    )

    print("\n" + "=" * 70)
    print("EMNIST MODEL COMPARISON")
    print("=" * 70)

    print(
        f"Parameter reduction : "
        f"{parameter_reduction:.2f}x"
    )

    print(
        f"Model size reduction : "
        f"{size_reduction:.2f}x"
    )

    print(
        f"Accuracy gap : "
        f"{accuracy_gap * 100:.2f} percentage points"
    )

    # Save results
    output_dir = Path("results/emnist")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_dir / "standard_confusion_matrix.npy",
        standard["confusion_matrix"]
    )

    np.save(
        output_dir / "lightweight_confusion_matrix.npy",
        lightweight["confusion_matrix"]
    )

    print(
        f"\nConfusion matrices saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()