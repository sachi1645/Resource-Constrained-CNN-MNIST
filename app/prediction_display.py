import tkinter as tk


class PredictionDisplay:
    """
    Displays the predicted digit and model confidence.
    """

    def __init__(self, parent):
        self.frame = tk.Frame(parent)
        self.frame.pack(pady=5)

        self.prediction_label = tk.Label(
            self.frame,
            text="Prediction: -",
            font=("Arial", 16, "bold")
        )
        self.prediction_label.pack(pady=5)

        self.confidence_label = tk.Label(
            self.frame,
            text="Confidence: -",
            font=("Arial", 14)
        )
        self.confidence_label.pack(pady=5)

    def update(self, digit, confidence):
        """
        Update the prediction and confidence displayed.
        """

        self.prediction_label.config(
            text=f"Prediction: {digit}"
        )

        self.confidence_label.config(
            text=f"Confidence: {confidence * 100:.2f}%"
        )

    def clear(self):
        """
        Reset the prediction display.
        """

        self.prediction_label.config(
            text="Prediction: -"
        )

        self.confidence_label.config(
            text="Confidence: -"
        )