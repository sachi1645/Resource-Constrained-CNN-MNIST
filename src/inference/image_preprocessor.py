from pathlib import Path
import sys

from PIL import Image

# Allows this file to be executed directly from the project root if needed.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import pil_to_tensor


def preprocess_image(image: Image.Image):
    return pil_to_tensor(image)
