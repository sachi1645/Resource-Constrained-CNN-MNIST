import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms

from src.data import dataset
from src.data import dataset
from src.inference.emnist_preprocessor import preprocess_emnist_image
from src.models.emnist_lightweight_cnn import EMNISTLightweightCNN

import torch


def tensor_to_pil(tensor):
    """Convert EMNIST tensor to PIL."""

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
    """Convert PIL image to EMNIST model format."""

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
    print("REAL EMNIST CHARACTER PIPELINE TEST")
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

    index = 0

    image_tensor, label = dataset[index]

    actual = dataset.classes[label]

    print(f"Actual character: {actual}")

    # ---------------------------------------------------------
    # Original EMNIST image
    # ---------------------------------------------------------

    original = tensor_to_pil(
        image_tensor
    )

    # ---------------------------------------------------------
    # Convert raw EMNIST g to upright orientation
    #
    # Our orientation search showed:
    #
    # Rotate 270 -> upright-looking character
    # ---------------------------------------------------------

    upright = original.rotate(
        270,
        expand=True
    )

    # ---------------------------------------------------------
    # Make a large 280x280 canvas
    #
    # This simulates the drawing application's canvas.
    # ---------------------------------------------------------

    drawing_canvas = Image.new(
        "L",
        (280, 280),
        0
    )

    # Resize upright EMNIST character
    # from 28x28 to 200x200.
    #
    # This simulates a character drawn on a large canvas.

    upright_large = upright.resize(
        (200, 200),
        Image.Resampling.NEAREST
    )

    # Center it
    x = (280 - 200) // 2
    y = (280 - 200) // 2

    drawing_canvas.paste(
        upright_large,
        (x, y)
    )

    # ---------------------------------------------------------
    # Apply the EXACT app preprocessing
    # ---------------------------------------------------------

    processed = preprocess_emnist_image(
        drawing_canvas
    )

    if processed is None:

        print(
            "ERROR: Character was not detected."
        )

        return

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
    # Predict
    # ---------------------------------------------------------

    with torch.no_grad():

        processed_device = processed.to(device)

        output = model(processed_device)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence_tensor, prediction_tensor = torch.max(
        probabilities,
        dim=1
    )

    prediction_index = prediction_tensor.item()

    prediction = dataset.classes[prediction_index]

    confidence = confidence_tensor.item()

    print("\nApp-style prediction:")
    print(
        f"Prediction: {prediction}"
    )
    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Prepare processed image for display
    # ---------------------------------------------------------

    processed_display = (
        processed.squeeze(0).squeeze(0)
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
        np.array(original),
        cmap="gray"
    )

    axes[0].set_title(
        f"Original EMNIST\nActual: {actual}"
    )

    axes[0].axis("off")

    axes[1].imshow(
        np.array(drawing_canvas),
        cmap="gray"
    )

    axes[1].set_title(
        "Simulated User Drawing"
    )

    axes[1].axis("off")

    axes[2].imshow(
        processed_display,
        cmap="gray"
    )

    axes[2].set_title(
        f"After App Preprocessing\n"
        f"Prediction: {prediction}"
    )

    axes[2].axis("off")

    plt.suptitle(
        "Real EMNIST → Simulated Drawing → App",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()