from PIL import Image
import torchvision.transforms as transforms


EMNIST_MEAN = 0.1307
EMNIST_STD = 0.3081


def preprocess_emnist_image(image):
    """
    Preprocess a character drawn on the application canvas.

    Pipeline:

        User drawing
            ↓
        Grayscale
            ↓
        Rotate 90 degrees
            ↓
        Resize complete canvas to 28x28
            ↓
        Normalize
            ↓
        EMNIST CNN
    """

    # ---------------------------------------------------------
    # 1. Convert to grayscale
    # ---------------------------------------------------------

    image = image.convert("L")

    # ---------------------------------------------------------
    # 2. Convert normal drawing orientation
    #    to EMNIST orientation
    # ---------------------------------------------------------

    image = image.rotate(
        90,
        expand=True
    )

    # ---------------------------------------------------------
    # 3. Resize the COMPLETE drawing canvas
    #
    # IMPORTANT:
    # Do NOT crop.
    # Do NOT resize the character separately.
    # ---------------------------------------------------------

    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # ---------------------------------------------------------
    # 4. Convert to tensor
    # ---------------------------------------------------------

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (EMNIST_MEAN,),
            (EMNIST_STD,)
        )
    ])

    tensor = transform(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor