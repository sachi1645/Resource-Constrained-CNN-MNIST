from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
])


def prepare_pil_digit(image: Image.Image) -> Image.Image:
    """Convert a user drawing into a centered 28x28 MNIST-like image."""
    image = image.convert("L")

    # Make sure foreground is white on a black background.
    arr = np.asarray(image)
    if arr.max() == 0:
        return Image.new("L", (28, 28), 0)

    # Threshold weak anti-aliased pixels.
    arr = np.where(arr > 20, arr, 0).astype(np.uint8)
    image = Image.fromarray(arr, mode="L")

    bbox = image.getbbox()
    if bbox is None:
        return Image.new("L", (28, 28), 0)

    image = image.crop(bbox)

    # MNIST digits are normally around 20x20 inside a 28x28 image.
    max_side = max(image.size)
    scale = 20.0 / max_side
    new_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)

    # Center the digit.
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))

    return canvas


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a prepared 28x28 PIL image into a normalized model tensor."""
    image = prepare_pil_digit(image)
    tensor = transforms.ToTensor()(image)
    tensor = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))(tensor)
    return tensor.unsqueeze(0)


def tensor_to_display_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor back to a viewable grayscale image."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3:
        tensor = tensor[0]

    tensor = tensor.detach().cpu()
    tensor = tensor * MNIST_STD + MNIST_MEAN
    tensor = tensor.clamp(0, 1)
    array = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")
