import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.inference.emnist_preprocessor import preprocess_emnist_image


def create_test_digit():
    """
    Create a simple handwritten-style digit image
    similar to what the drawing canvas produces.
    """

    image = Image.new("L", (280, 280), 0)

    # Draw a simple digit 5
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)

    draw.line(
        [(180, 40), (80, 40), (80, 130)],
        fill=255,
        width=18
    )

    draw.line(
        [(80, 130), (170, 130)],
        fill=255,
        width=18
    )

    draw.line(
        [(170, 130), (170, 230)],
        fill=255,
        width=18
    )

    draw.line(
        [(170, 230), (80, 230)],
        fill=255,
        width=18
    )

    return image


def main():

    print("=" * 60)
    print("EMNIST PREPROCESSING VISUAL TEST")
    print("=" * 60)

    # Create test drawing
    original_image = create_test_digit()

    # Apply our EMNIST preprocessing
    processed_tensor = preprocess_emnist_image(
        original_image
    )

    if processed_tensor is None:
        print("ERROR: No character detected.")
        return

    # Convert tensor back to displayable image
    processed_image = processed_tensor.squeeze(0).squeeze(0)

    # Undo normalization
    processed_image = (
        processed_image * 0.3081
    ) + 0.1307

    processed_image = processed_image.clamp(
        0,
        1
    )

    processed_image = processed_image.numpy()

    # Display original and processed images
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    # Original
    axes[0].imshow(
        np.array(original_image),
        cmap="gray"
    )

    axes[0].set_title(
        "Original Drawing"
    )

    axes[0].axis("off")

    # Processed
    axes[1].imshow(
        processed_image,
        cmap="gray"
    )

    axes[1].set_title(
        "After EMNIST Preprocessing"
    )

    axes[1].axis("off")

    plt.suptitle(
        "EMNIST Preprocessing Check",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()