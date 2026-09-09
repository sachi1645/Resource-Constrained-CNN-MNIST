from pathlib import Path

import torch

from src.data.preprocessing import pil_to_tensor
from src.models.lightweight_cnn import LightweightCNN
from src.models.model_utils import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "lightweight_cnn" / "best_model.pth"


class DigitPredictor:
    def __init__(self, model_path=DEFAULT_MODEL, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = LightweightCNN()
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Train the lightweight model first."
            )

        load_checkpoint(self.model, model_path, self.device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        tensor = pil_to_tensor(image).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)

        return int(prediction.item()), float(confidence.item()), probabilities[0].cpu()


if __name__ == "__main__":
    print("DigitPredictor is ready after a trained lightweight model exists.")
