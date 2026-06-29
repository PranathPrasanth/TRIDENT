"""
TRIDENT Grad-CAM Visualizer

Visualizes Grad-CAM heatmaps over
Mel Spectrograms.
"""

import matplotlib.pyplot as plt
import numpy as np

from utils.logger import logger
from utils.helpers import save_figure


class GradCAMVisualizer:
    """
    Visualizes Grad-CAM heatmaps.
    """

    def display(
        self,
        spectrogram: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> None:
        """
        Display the Grad-CAM heatmap over
        the Mel Spectrogram.
        """

        logger.info("Displaying Grad-CAM...")

        plt.figure(figsize=(10, 6))

        plt.imshow(
            spectrogram,
            cmap="gray",
            origin="lower",
            aspect="auto",
        )

        plt.imshow(
            heatmap,
            cmap="jet",
            alpha=alpha,
            origin="lower",
            aspect="auto",
        )

        plt.title("TRIDENT Grad-CAM Explanation")

        plt.xlabel("Time")

        plt.ylabel("Mel Frequency")

        plt.colorbar(label="Importance")

        plt.tight_layout()

        plt.show()

        logger.info("Visualization completed.")

    # ---------------------------------------------------------

    def save(
        self,
        spectrogram: np.ndarray,
        heatmap: np.ndarray,
        output_path: str,
        alpha: float = 0.4,
    ) -> None:
        """
        Save the Grad-CAM visualization.
        """

        logger.info("Saving Grad-CAM visualization...")

        plt.figure(figsize=(10, 6))

        plt.imshow(
            spectrogram,
            cmap="gray",
            origin="lower",
            aspect="auto",
        )

        plt.imshow(
            heatmap,
            cmap="jet",
            alpha=alpha,
            origin="lower",
            aspect="auto",
        )

        plt.title("TRIDENT Grad-CAM Explanation")

        plt.xlabel("Time")

        plt.ylabel("Mel Frequency")

        plt.tight_layout()

        save_figure(output_path)

        plt.close()

        logger.info("Visualization saved.")


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    print("GradCAMVisualizer module loaded successfully.")