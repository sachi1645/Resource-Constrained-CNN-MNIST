import tkinter as tk
from tkinter import messagebox

from app.drawing_canvas import DrawingCanvas
from src.inference.predict import DigitPredictor


class MNISTApp:
    def __init__(self, root):
        self.root = root

        self.root.title("MNIST Digit Recognition")
        self.root.geometry("420x500")
        self.root.resizable(False, False)

        # Title
        title = tk.Label(
            root,
            text="MNIST Digit Recognition",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=15)

        # Instructions
        instructions = tk.Label(
            root,
            text="Draw a digit using your mouse",
            font=("Arial", 11)
        )
        instructions.pack(pady=5)

        # Drawing canvas
        self.drawing = DrawingCanvas(
            root,
            width=280,
            height=280
        )

        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=15)

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            width=12,
            command=self.clear
        )
        clear_button.grid(row=0, column=0, padx=10)

        predict_button = tk.Button(
            button_frame,
            text="Predict",
            width=12,
            command=self.predict
        )
        predict_button.grid(row=0, column=1, padx=10)

        # Prediction label
        self.prediction_label = tk.Label(
            root,
            text="Prediction: -",
            font=("Arial", 16, "bold")
        )
        self.prediction_label.pack(pady=5)

        # Confidence label
        self.confidence_label = tk.Label(
            root,
            text="Confidence: -",
            font=("Arial", 14)
        )
        self.confidence_label.pack(pady=5)

        # Load trained lightweight model
        try:
            self.predictor = DigitPredictor()
        except Exception as error:
            self.predictor = None

            messagebox.showerror(
                "Model Error",
                f"Could not load the trained model.\n\n{error}"
            )

    def clear(self):
        self.drawing.clear()

        self.prediction_label.config(
            text="Prediction: -"
        )

        self.confidence_label.config(
            text="Confidence: -"
        )

    def predict(self):
        if self.predictor is None:
            messagebox.showerror(
                "Prediction Error",
                "The trained model could not be loaded."
            )
            return

        try:
            image = self.drawing.get_image()

            digit, confidence, probabilities = (
                self.predictor.predict(image)
            )

            self.prediction_label.config(
                text=f"Prediction: {digit}"
            )

            self.confidence_label.config(
                text=f"Confidence: {confidence * 100:.2f}%"
            )

        except Exception as error:
            messagebox.showerror(
                "Prediction Error",
                str(error)
            )


def main():
    root = tk.Tk()

    app = MNISTApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()