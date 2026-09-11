import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import datasets, transforms

from src.inference.emnist_preprocessor import preprocess_emnist_image


def tensor_to_pil(tensor):
    """Convert a [1, 28, 28] tensor to a PIL grayscale image."""

    image = tensor.squeeze(0).numpy()

    image = (image * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(image, mode="L")


def main():

    print("=" * 60)
    print("EMNIST TRANSFORMATION TEST")
    print("=" * 60)

    # Load EMNIST exactly as the training dataset
    dataset = datasets.EMNIST(
        root="data/raw",
        split="balanced",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # Select one sample
    index = 0

    image_tensor, label = dataset[index]

    character = dataset.classes[label]

    print(f"\nSample index: {index}")
    print(f"Label index: {label}")
    print(f"Character: {character}")

    # ---------------------------------------------------------
    # 1. Original EMNIST image
    # ---------------------------------------------------------

    emnist_image = tensor_to_pil(image_tensor)

    # ---------------------------------------------------------
    # 2. Convert EMNIST orientation to normal human orientation
    # ---------------------------------------------------------
    #
    # Our preprocessing does:
    #
    # rotate 90 degrees
    # then flip horizontally
    #
    # Therefore we use the inverse operation here.
    #

    normal_image = emnist_image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    normal_image = normal_image.rotate(
        270,
        expand=True
    )

    # ---------------------------------------------------------
    # 3. Send the normal image through our preprocessing
    # ---------------------------------------------------------

    processed_tensor = preprocess_emnist_image(
        normal_image
    )

    if processed_tensor is None:
        print("ERROR: Character was not detected.")
        return

    # Remove batch and channel dimensions
    processed_image = processed_tensor.squeeze(
        0
    ).squeeze(
        0
    )

    # Undo normalization for display
    processed_image = (
        processed_image * 0.3081
    ) + 0.1307

    processed_image = processed_image.clamp(
        0,
        1
    )

    processed_image = (
        processed_image.numpy() * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    # Original EMNIST
    axes[0].imshow(
        np.array(emnist_image),
        cmap="gray"
    )

    axes[0].set_title(
        f"Original EMNIST\nLabel: {character}"
    )

    axes[0].axis("off")

    # Normal orientation
    axes[1].imshow(
        np.array(normal_image),
        cmap="gray"
    )

    axes[1].set_title(
        "Normal Orientation"
    )

    axes[1].axis("off")

    # After app preprocessing
    axes[2].imshow(
        processed_image,
        cmap="gray"
    )

    axes[2].set_title(
        "After App Preprocessing"
    )

    axes[2].axis("off")

    plt.suptitle(
        "EMNIST Orientation Transformation Test",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()