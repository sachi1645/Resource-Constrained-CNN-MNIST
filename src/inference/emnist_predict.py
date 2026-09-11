from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets

from src.inference.emnist_preprocessor import preprocess_emnist_image
from src.models.emnist_standard_cnn import EMNISTStandardCNN
from src.models.emnist_lightweight_cnn import EMNISTLightweightCNN


class EMNISTPredictor:

    def __init__(self, model_type="lightweight"):

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # --------------------------------------------------
        # Load EMNIST class names directly from torchvision
        # --------------------------------------------------

        class_dataset = datasets.EMNIST(
            root="data/raw",
            split="balanced",
            train=False,
            download=True
        )

        self.classes = class_dataset.classes

        print("EMNIST classes:")
        print(self.classes)

        # --------------------------------------------------
        # Select model
        # --------------------------------------------------

        if model_type == "standard":

            self.model = EMNISTStandardCNN(
                num_classes=47
            )

            model_path = Path(
                "models/emnist_standard_cnn/best_model.pth"
            )

        elif model_type == "lightweight":

            self.model = EMNISTLightweightCNN(
                num_classes=47
            )

            model_path = Path(
                "models/emnist_lightweight_cnn/best_model.pth"
            )

        else:

            raise ValueError(
                "model_type must be 'standard' or 'lightweight'"
            )

        # --------------------------------------------------
        # Check model file
        # --------------------------------------------------

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        # --------------------------------------------------
        # Load trained weights
        # --------------------------------------------------

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            self.model.load_state_dict(
                checkpoint
            )

        # --------------------------------------------------
        # Prepare model
        # --------------------------------------------------

        self.model.to(self.device)

        self.model.eval()

        print(
            f"EMNIST {model_type} model loaded successfully."
        )

        print(
            f"Device: {self.device}"
        )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    def predict(self, image: Image.Image):

        # Preprocess user image
        tensor = preprocess_emnist_image(
            image
        )

        if tensor is None:

            raise ValueError(
                "No character was detected in the drawing."
            )

        # Move image to GPU / CPU
        tensor = tensor.to(
            self.device
        )

        # --------------------------------------------------
        # Model prediction
        # --------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                tensor
            )

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        # --------------------------------------------------
        # Convert class index to character
        # --------------------------------------------------

        predicted_index = prediction.item()

        predicted_character = self.classes[
            predicted_index
        ]

        confidence_value = confidence.item()

        return (
            predicted_character,
            confidence_value,
            probabilities[0].cpu().numpy()
        )