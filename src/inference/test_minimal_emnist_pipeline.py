import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms

import torch

from src.models.emnist_lightweight_cnn import EMNISTLightweightCNN


def tensor_to_pil(tensor):
    image = tensor.squeeze(0).numpy()

    image = (
        image * 255
    ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(image, mode="L")


def predict(model, image, device, classes):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = prediction.item()

    return (
        classes[predicted_index],
        confidence.item()
    )


def main():

    print("=" * 70)
    print("MINIMAL EMNIST APP PIPELINE TEST")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    # ---------------------------------------------------------
    # Load EMNIST
    # ---------------------------------------------------------

    dataset = datasets.EMNIST(
        root="data/raw",
        split="balanced",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    image_tensor, label = dataset[0]

    actual = dataset.classes[label]

    print(f"Actual character: {actual}")

    # ---------------------------------------------------------
    # Original EMNIST
    # ---------------------------------------------------------

    original = tensor_to_pil(
        image_tensor
    )

    # ---------------------------------------------------------
    # Create simulated upright drawing
    # ---------------------------------------------------------

    upright = original.rotate(
        270,
        expand=True
    )

    canvas = Image.new(
        "L",
        (280, 280),
        0
    )

    large = upright.resize(
        (200, 200),
        Image.Resampling.LANCZOS
    )

    x = (280 - 200) // 2
    y = (280 - 200) // 2

    canvas.paste(
        large,
        (x, y)
    )

    # ---------------------------------------------------------
    # Minimal app preprocessing
    # ---------------------------------------------------------
    #
    # 1. Rotate user drawing 90 degrees
    # 2. Resize the COMPLETE canvas to 28x28
    #
    # NO crop
    # NO additional centering
    # NO resize-to-20
    #

    processed = canvas.rotate(
        90,
        expand=True
    )

    processed = processed.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    prediction, confidence = predict(
        model=load_model(device),
        image=processed,
        device=device,
        classes=dataset.classes
    )

    print("\nMinimal pipeline prediction:")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence * 100:.2f}%")

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(
        np.array(original),
        cmap="gray"
    )

    axes[0].set_title(
        f"Original EMNIST\n{actual}"
    )

    axes[0].axis("off")

    axes[1].imshow(
        np.array(canvas),
        cmap="gray"
    )

    axes[1].set_title(
        "Simulated User Drawing"
    )

    axes[1].axis("off")

    axes[2].imshow(
        np.array(processed),
        cmap="gray"
    )

    axes[2].set_title(
        f"Minimal Preprocessing\n"
        f"Prediction: {prediction}"
    )

    axes[2].axis("off")

    plt.suptitle(
        "Minimal EMNIST App Pipeline",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


def load_model(device):

    model = EMNISTLightweightCNN(
        num_classes=47
    )

    checkpoint = torch.load(
        "models/emnist_lightweight_cnn/best_model.pth",
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    main()