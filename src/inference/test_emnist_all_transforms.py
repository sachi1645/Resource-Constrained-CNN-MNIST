import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms


def tensor_to_pil(tensor):
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


def main():

    print("=" * 70)
    print("EMNIST ALL GEOMETRIC TRANSFORMATIONS")
    print("=" * 70)

    dataset = datasets.EMNIST(
        root="data/raw",
        split="balanced",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    image_tensor, label = dataset[0]

    actual = dataset.classes[label]

    image = tensor_to_pil(image_tensor)

    print(f"\nActual character: {actual}")

    # ---------------------------------------------------------
    # Generate transformations
    # ---------------------------------------------------------

    transformations = []

    # 1. Original
    transformations.append(
        ("Original", image)
    )

    # 2. Rotate 90
    transformations.append(
        ("Rotate 90", image.rotate(90, expand=True))
    )

    # 3. Rotate 180
    transformations.append(
        ("Rotate 180", image.rotate(180, expand=True))
    )

    # 4. Rotate 270
    transformations.append(
        ("Rotate 270", image.rotate(270, expand=True))
    )

    # 5. Horizontal flip
    transformations.append(
        (
            "Horizontal Flip",
            image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
        )
    )

    # 6. Vertical flip
    transformations.append(
        (
            "Vertical Flip",
            image.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )
        )
    )

    # 7. Transpose
    transformations.append(
        (
            "Transpose",
            image.transpose(
                Image.Transpose.TRANSPOSE
            )
        )
    )

    # 8. Transverse
    transformations.append(
        (
            "Transverse",
            image.transpose(
                Image.Transpose.TRANSVERSE
            )
        )
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 7)
    )

    for ax, (name, transformed) in zip(
        axes.flat,
        transformations
    ):

        ax.imshow(
            np.array(transformed),
            cmap="gray"
        )

        ax.set_title(name)

        ax.axis("off")

    plt.suptitle(
        f"EMNIST Transformations - Actual: {actual}",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()