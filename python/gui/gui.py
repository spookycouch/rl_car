from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> Dict[int, Tuple[int, int]]:
        """Detect points of interest in an image."""


class BaseSelector(Detector):
    """Base class for GUI-based selectors."""

    def __init__(self):
        self.root = None
        self.selection = None
        self._setup_gui()

    def _setup_gui(self):
        self.root = Tk()
        self.root.title(self._get_window_title())

        # Main frame
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=BOTH, expand=True)

        # Canvas for image
        self.canvas = Canvas(self.frame)
        self.canvas.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Mouse bindings
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # Drawing state
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_shape = None

        # Store reference to photo to prevent garbage collection
        self.photo = None

    @abstractmethod
    def _get_window_title(self) -> str:
        """Return the window title for this selector."""
        pass

    @abstractmethod
    def _on_mouse_down(self, event):
        """Handle mouse down event."""
        pass

    @abstractmethod
    def _on_mouse_move(self, event):
        """Handle mouse move event."""
        pass

    @abstractmethod
    def _on_mouse_up(self, event):
        """Handle mouse up event."""
        pass

    @abstractmethod
    def _get_points(self) -> Dict[int, Tuple[int, int]]:
        """Convert the current selection to point dictionary."""
        pass

    def detect(self, image: np.ndarray) -> Dict[int, Tuple[int, int]]:
        """Display image and let user make selection, return points."""
        # Reset state
        self.selection = None

        height, width = image.shape[:2]

        # Create a copy for drawing
        display = image.copy()

        # Add instructions
        instructions = self._get_instructions()
        cv2.putText(
            display,
            instructions,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Convert image for display
        if display.shape[-1] == 3:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        else:
            display = display

        # Update canvas size
        height, width = display.shape[:2]
        self.canvas.config(width=width, height=height)

        # Show image
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(display))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        # Wait for user selection
        self.root.wait_window(self.root)

        # Return points based on selection
        return self._get_points() if self.selection else {}

    @abstractmethod
    def _get_instructions(self) -> str:
        """Return instructions for this selector type."""
        pass


class PointSelector(BaseSelector):
    """Selector for single point selection."""

    def _get_window_title(self) -> str:
        return "Point Selector"

    def _get_instructions(self) -> str:
        return "Click to select a single point"

    def _on_mouse_down(self, event):
        self.selection = (event.x, event.y)
        if self.current_shape:
            self.canvas.delete(self.current_shape)
        self.current_shape = self.canvas.create_oval(
            event.x - 3,
            event.y - 3,
            event.x + 3,
            event.y + 3,
            fill="red",
            outline="red",
        )
        self.root.destroy()

    def _on_mouse_move(self, event):
        pass  # No movement needed for point selection

    def _on_mouse_up(self, event):
        pass  # Selection already handled in mouse down

    def _get_points(self) -> Dict[int, Tuple[int, int]]:
        return {0: self.selection} if self.selection else {}


class BoundingBoxSelector(BaseSelector):
    """Selector for bounding box selection."""

    def _get_window_title(self) -> str:
        return "Bounding Box Selector"

    def _get_instructions(self) -> str:
        return "Click and drag to draw a bounding box"

    def _on_mouse_down(self, event):
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y

    def _on_mouse_move(self, event):
        if self.drawing:
            if self.current_shape:
                self.canvas.delete(self.current_shape)
            self.current_shape = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y, outline="green", width=2
            )

    def _on_mouse_up(self, event):
        if self.drawing:
            self.drawing = False
            w = event.x - self.start_x
            h = event.y - self.start_y
            self.selection = (self.start_x, self.start_y, w, h)
            self.root.destroy()

    def _get_points(self) -> Dict[int, Tuple[int, int]]:
        if not self.selection:
            return {}
        x, y, w, h = self.selection
        return {
            0: (x, y),  # Top-left
            1: (x + w, y),  # Top-right
            2: (x, y + h),  # Bottom-left
            3: (x + w, y + h),  # Bottom-right
        }


def main():
    # Example usage of both selectors
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(
        image,
        "Click to select",
        (50, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    # Point selection example
    point_selector = PointSelector()
    point = point_selector.detect(image)
    print("Selected point:", point)

    # Bounding box selection example
    # bbox_selector = BoundingBoxSelector()
    # corners = bbox_selector.detect(image)
    # print("Bounding box corners:", corners)


if __name__ == "__main__":
    main()
