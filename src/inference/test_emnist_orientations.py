import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms

from src.models.emnist_lightweight_cnn import EMNISTLightweightCNN


def tensor_to_pil(tensor):
    """Convert [1, 28, 28] tensor to PIL image."""

    image = tensor.squeeze(0).numpy()

    image = (
        image * 255
    ).clip(
        0, 255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        image,
        mode="L"
    )


def pil_to_tensor(image):
    """Convert PIL image to the same format used during training."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])

    return transform(image).unsqueeze(0)


def predict(model, image, device, classes):

    tensor = pil_to_tensor(image).to(device)

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

    index = prediction.item()

    return (
        classes[index],
        confidence.item()
    )


def main():

    print("=" * 70)
    print("EMNIST ORIENTATION SEARCH")
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

    # Use sample 0
    index = 0

    image_tensor, label = dataset[index]

    actual = dataset.classes[label]

    print(f"\nSample index: {index}")
    print(f"Actual character: {actual}")

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Original image
    # ---------------------------------------------------------

    original = tensor_to_pil(
        image_tensor
    )

    # ---------------------------------------------------------
    # Generate all orientations
    # ---------------------------------------------------------

    images = []

    current = original

    for rotation in [0, 90, 180, 270]:

        rotated = original.rotate(
            rotation,
            expand=True
        )

        # Without flip
        images.append(
            (
                f"Rotate {rotation}",
                rotated
            )
        )

        # With horizontal flip
        flipped = rotated.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        images.append(
            (
                f"Rotate {rotation} + Flip",
                flipped
            )
        )

    # ---------------------------------------------------------
    # Test every orientation
    # ---------------------------------------------------------

    print("\nOrientation predictions:\n")

    results = []

    for name, image in images:

        prediction, confidence = predict(
            model,
            image,
            device,
            dataset.classes
        )

        results.append(
            (
                name,
                image,
                prediction,
                confidence
            )
        )

        print(
            f"{name:<25} "
            f"Prediction = {prediction:<3} "
            f"Confidence = {confidence * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Display all orientations
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 7)
    )

    for ax, (
        name,
        image,
        prediction,
        confidence
    ) in zip(
        axes.flat,
        results
    ):

        ax.imshow(
            np.array(image),
            cmap="gray"
        )

        ax.set_title(
            f"{name}\n"
            f"Prediction: {prediction}\n"
            f"{confidence * 100:.1f}%"
        )

        ax.axis("off")

    plt.suptitle(
        f"EMNIST Orientation Search - Actual: {actual}",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()