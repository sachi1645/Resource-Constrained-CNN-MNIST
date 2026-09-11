import numpy as np
from PIL import Image
from torchvision import datasets, transforms

from src.inference.emnist_predict import EMNISTPredictor


def tensor_to_pil(tensor):
    """
    Convert an EMNIST [1, 28, 28] tensor
    into a PIL grayscale image.
    """

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


def convert_to_normal_orientation(image):
    """
    Convert the EMNIST stored orientation
    into normal human-readable orientation.
    """

    # Undo the transformation used by
    # emnist_preprocessor.py

    image = image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    image = image.rotate(
        270,
        expand=True
    )

    return image


def main():

    print("=" * 60)
    print("EMNIST APP-STYLE INFERENCE TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Load EMNIST test dataset
    # --------------------------------------------------

    test_dataset = datasets.EMNIST(
        root="data/raw",
        split="balanced",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # --------------------------------------------------
    # Load our predictor
    # --------------------------------------------------

    predictor = EMNISTPredictor(
        model_type="lightweight"
    )

    # --------------------------------------------------
    # Samples to test
    # --------------------------------------------------

    test_indices = [
        0,
        100,
        500,
        1000,
        5000
    ]

    correct = 0

    print("\nTesting samples...\n")

    for index in test_indices:

        # Get EMNIST image
        image_tensor, label = test_dataset[index]

        actual_character = test_dataset.classes[label]

        # Convert tensor → PIL
        emnist_image = tensor_to_pil(
            image_tensor
        )

        # Convert EMNIST image to normal orientation
        normal_image = convert_to_normal_orientation(
            emnist_image
        )

        # --------------------------------------------------
        # Use exactly the same predictor used by the app
        # --------------------------------------------------

        predicted_character, confidence, _ = (
            predictor.predict(
                normal_image
            )
        )

        # Check result
        if predicted_character == actual_character:
            correct += 1
            result = "CORRECT"
        else:
            result = "WRONG"

        print(
            f"Sample {index}: "
            f"Actual = {actual_character}, "
            f"Predicted = {predicted_character}, "
            f"Confidence = {confidence * 100:.2f}%, "
            f"{result}"
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    accuracy = (
        correct / len(test_indices)
    ) * 100

    print("\n" + "=" * 60)

    print(
        f"Correct: {correct}/{len(test_indices)}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()