import tkinter as tk
from PIL import Image, ImageGrab


class DrawingCanvas:
    def __init__(self, parent, width=280, height=280):
        self.width = width
        self.height = height

        self.canvas = tk.Canvas(
            parent,
            width=width,
            height=height,
            bg="black",
            highlightthickness=2,
            highlightbackground="gray"
        )

        self.canvas.pack()

        self.last_x = None
        self.last_y = None

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill="white",
                width=18,
                capstyle=tk.ROUND,
                smooth=True
            )

        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete("all")

    def get_image(self):
        """
        Capture the drawing canvas as a PIL image.
        """

        self.canvas.update()

        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()

        x1 = x + self.width
        y1 = y + self.height

        image = ImageGrab.grab(
            bbox=(x, y, x1, y1)
        )

        return image.convert("L")

if __name__ == "__main__":
    import tkinter as tk

    root = tk.Tk()
    root.title("MNIST Drawing Test")

    drawing = DrawingCanvas(root)

    clear_button = tk.Button(
        root,
        text="Clear",
        command=drawing.clear
    )
    clear_button.pack(pady=10)

    root.mainloop()