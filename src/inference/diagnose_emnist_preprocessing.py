import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms

from src.inference.emnist_preprocessor import preprocess_emnist_image
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


def emnist_to_normal(image):
    """
    Convert EMNIST orientation to normal orientation.

    The app preprocessing does:

        normal
          ↓
        FLIP_LEFT_RIGHT
          ↓
        TRANSPOSE
          ↓
        EMNIST

    Therefore the inverse is:

        EMNIST
          ↓
        TRANSPOSE
          ↓
        FLIP_LEFT_RIGHT
          ↓
        normal
    """

    image = image.transpose(
        Image.Transpose.TRANSPOSE
    )

    image = image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    return image


def pil_to_normalized_tensor(image):
    """Convert PIL image to EMNIST training format."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])

    return transform(image).unsqueeze(0)


def predict(model, tensor, device, classes):

    tensor = tensor.to(device)

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
    print("EMNIST PREPROCESSING DIAGNOSTIC")
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

    # Sample
    index = 0

    original_tensor, label = dataset[index]

    actual = dataset.classes[label]

    print(f"\nSample: {index}")
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
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # TEST 1
    # Original EMNIST tensor
    # ---------------------------------------------------------

    original_normalized = (
        (original_tensor - 0.1307)
        / 0.3081
    ).unsqueeze(0)

    prediction1, confidence1 = predict(
        model,
        original_normalized,
        device,
        dataset.classes
    )

    print("\nTEST 1 - Original EMNIST")
    print(
        f"Prediction: {prediction1}"
    )
    print(
        f"Confidence: {confidence1 * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Convert EMNIST image to PIL
    # ---------------------------------------------------------

    original_pil = tensor_to_pil(
        original_tensor
    )

    # ---------------------------------------------------------
    # Convert EMNIST → normal
    # ---------------------------------------------------------

    normal_pil = emnist_to_normal(
        original_pil
    )

    # ---------------------------------------------------------
    # TEST 2
    # Normal orientation without app preprocessing
    # ---------------------------------------------------------

    normal_tensor = pil_to_normalized_tensor(
        normal_pil
    )

    prediction2, confidence2 = predict(
        model,
        normal_tensor,
        device,
        dataset.classes
    )

    print("\nTEST 2 - Normal orientation only")
    print(
        f"Prediction: {prediction2}"
    )
    print(
        f"Confidence: {confidence2 * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # TEST 3
    # Full app preprocessing
    # ---------------------------------------------------------

    processed_tensor = preprocess_emnist_image(
        normal_pil
    )

    if processed_tensor is None:

        print(
            "\nERROR: Character was not detected."
        )

        return

    prediction3, confidence3 = predict(
        model,
        processed_tensor,
        device,
        dataset.classes
    )

    print("\nTEST 3 - Full app preprocessing")
    print(
        f"Prediction: {prediction3}"
    )
    print(
        f"Confidence: {confidence3 * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Tensor difference
    # ---------------------------------------------------------

    difference = torch.mean(
        torch.abs(
            original_normalized
            - processed_tensor
        )
    ).item()

    print("\nTensor difference:")
    print(
        f"Mean absolute difference: {difference:.6f}"
    )

    # ---------------------------------------------------------
    # Prepare processed image for display
    # ---------------------------------------------------------

    processed_display = (
        processed_tensor
        .squeeze(0)
        .squeeze(0)
    )

    processed_display = (
        processed_display * 0.3081
    ) + 0.1307

    processed_display = (
        processed_display
        .clamp(0, 1)
        .numpy()
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(
        np.array(original_pil),
        cmap="gray"
    )

    axes[0].set_title(
        f"Original EMNIST\n{actual}"
    )

    axes[0].axis("off")

    axes[1].imshow(
        np.array(normal_pil),
        cmap="gray"
    )

    axes[1].set_title(
        f"Normal Orientation\n{prediction2}"
    )

    axes[1].axis("off")

    axes[2].imshow(
        processed_display,
        cmap="gray"
    )

    axes[2].set_title(
        f"App Preprocessing\n{prediction3}"
    )

    axes[2].axis("off")

    plt.suptitle(
        "EMNIST Preprocessing Diagnostic",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()