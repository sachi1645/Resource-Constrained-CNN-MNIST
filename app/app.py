import tkinter as tk
from tkinter import messagebox

from app.drawing_canvas import DrawingCanvas
from src.inference.emnist_predict import EMNISTPredictor


class EMNISTApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "EMNIST Character Recognition"
        )

        self.root.geometry(
            "420x550"
        )

        self.root.resizable(
            False,
            False
        )

        # -----------------------------------------------------
        # Title
        # -----------------------------------------------------

        title = tk.Label(
            root,
            text="EMNIST Character Recognition",
            font=("Arial", 20, "bold")
        )

        title.pack(
            pady=15
        )

        # -----------------------------------------------------
        # Instructions
        # -----------------------------------------------------

        instructions = tk.Label(
            root,
            text="Draw a digit or character using your mouse",
            font=("Arial", 11)
        )

        instructions.pack(
            pady=5
        )

        # -----------------------------------------------------
        # Drawing canvas
        # -----------------------------------------------------

        self.drawing = DrawingCanvas(
            root,
            width=280,
            height=280
        )

        # -----------------------------------------------------
        # Model selection
        # -----------------------------------------------------

        model_frame = tk.Frame(root)

        model_frame.pack(
            pady=10
        )

        model_label = tk.Label(
            model_frame,
            text="Model:"
        )

        model_label.grid(
            row=0,
            column=0,
            padx=5
        )

        self.model_var = tk.StringVar(
            value="lightweight"
        )

        model_menu = tk.OptionMenu(
            model_frame,
            self.model_var,
            "lightweight",
            "standard"
        )

        model_menu.config(
            width=12
        )

        model_menu.grid(
            row=0,
            column=1,
            padx=5
        )

        # -----------------------------------------------------
        # Buttons
        # -----------------------------------------------------

        button_frame = tk.Frame(root)

        button_frame.pack(
            pady=10
        )

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            width=12,
            command=self.clear
        )

        clear_button.grid(
            row=0,
            column=0,
            padx=10
        )

        predict_button = tk.Button(
            button_frame,
            text="Predict",
            width=12,
            command=self.predict
        )

        predict_button.grid(
            row=0,
            column=1,
            padx=10
        )

        # -----------------------------------------------------
        # Prediction labels
        # -----------------------------------------------------

        self.prediction_label = tk.Label(
            root,
            text="Prediction: -",
            font=("Arial", 18, "bold")
        )

        self.prediction_label.pack(
            pady=5
        )

        self.confidence_label = tk.Label(
            root,
            text="Confidence: -",
            font=("Arial", 14)
        )

        self.confidence_label.pack(
            pady=5
        )

        # -----------------------------------------------------
        # Predictor
        # -----------------------------------------------------

        try:

            self.predictor = EMNISTPredictor(
                model_type="lightweight"
            )

        except Exception as error:

            self.predictor = None

            messagebox.showerror(
                "Model Error",
                f"Could not load the EMNIST model.\n\n{error}"
            )

    # =========================================================
    # Clear
    # =========================================================

    def clear(self):

        self.drawing.clear()

        self.prediction_label.config(
            text="Prediction: -"
        )

        self.confidence_label.config(
            text="Confidence: -"
        )

    # =========================================================
    # Predict
    # =========================================================

    def predict(self):

        if self.predictor is None:

            messagebox.showerror(
                "Prediction Error",
                "The EMNIST model could not be loaded."
            )

            return

        try:

            # Get drawing from canvas
            image = self.drawing.get_image()

            # Get selected model
            selected_model = self.model_var.get()

            # Load selected model
            self.predictor = EMNISTPredictor(
                model_type=selected_model
            )

            # Predict
            character, confidence, probabilities = (
                self.predictor.predict(image)
            )

            # Display prediction
            self.prediction_label.config(
                text=f"Prediction: {character}"
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

    app = EMNISTApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":

    main()