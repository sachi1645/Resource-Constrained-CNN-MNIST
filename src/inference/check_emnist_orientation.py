import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def main():
    # Load EMNIST Balanced test dataset
    test_dataset = datasets.EMNIST(
        root="data/raw",
        split="balanced",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    print("=" * 60)
    print("EMNIST ORIENTATION CHECK")
    print("=" * 60)

    print("\nEMNIST classes:")
    print(test_dataset.classes)

    print("\nFirst 10 labels:")
    for i in range(10):
        image, label = test_dataset[i]

        print(
            f"Sample {i}: "
            f"label index={label}, "
            f"class={test_dataset.classes[label]}"
        )

    # Display first 10 images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i, ax in enumerate(axes.flat):
        image, label = test_dataset[i]

        ax.imshow(image.squeeze(), cmap="gray")

        ax.set_title(
            f"Label: {test_dataset.classes[label]}"
        )

        ax.axis("off")

    plt.suptitle(
        "EMNIST Balanced - Orientation Check",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()