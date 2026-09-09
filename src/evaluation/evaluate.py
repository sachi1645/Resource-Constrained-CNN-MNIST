import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix

from src.data.dataset import get_dataloaders
from src.models.lightweight_cnn import LightweightCNN
from src.models.standard_cnn import StandardCNN
from src.models.model_utils import load_checkpoint, count_parameters, model_size_mb
from src.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[2]


def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    start = time.perf_counter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    elapsed = time.perf_counter() - start

    accuracy = correct / total
    avg_ms = (elapsed / total) * 1000

    return {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "average_inference_ms_per_image": avg_ms,
        "targets": np.array(all_targets),
        "predictions": np.array(all_predictions),
        "confusion_matrix": confusion_matrix(
            all_targets, all_predictions
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["standard", "lightweight"],
        default="lightweight",
    )
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "standard":
        model = StandardCNN()
        model_path = ROOT / "models" / "standard_cnn" / "best_model.pth"
    else:
        model = LightweightCNN()
        model_path = ROOT / "models" / "lightweight_cnn" / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Train it first."
        )

    load_checkpoint(model, model_path, device)
    model.to(device)

    _, _, test_loader = get_dataloaders(
        batch_size=256,
        num_workers=0,
        seed=42,
    )

    result = evaluate_model(model, test_loader, device)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Model size: {model_size_mb(model):.4f} MB")
    print(f"Test loss: {result['loss']:.4f}")
    print(f"Test accuracy: {result['accuracy'] * 100:.2f}%")
    print(
        f"Average inference time: "
        f"{result['average_inference_ms_per_image']:.4f} ms/image"
    )

    print("\nConfusion Matrix:")
    print(result["confusion_matrix"])


if __name__ == "__main__":
    main()
