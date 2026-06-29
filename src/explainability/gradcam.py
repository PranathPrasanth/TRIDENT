"""
TRIDENT Grad-CAM Generator

Generates Grad-CAM heatmaps for explaining
CNN predictions.
"""

import tensorflow as tf
import numpy as np

from utils.logger import logger


class GradCAM:
    """
    Generates Grad-CAM heatmaps.
    """

    def __init__(
        self,
        model: tf.keras.Model,
    ) -> None:

        self.model = model

    # ---------------------------------------------------------

    def generate(
        self,
        image: np.ndarray,
        last_conv_layer_name: str,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.
        """

        logger.info("Generating Grad-CAM heatmap...")

        grad_model = tf.keras.models.Model(

            inputs=self.model.inputs,

            outputs=[

                self.model.get_layer(
                    last_conv_layer_name
                ).output,

                self.model.output,

            ],

        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(
                image
            )

            predicted_class = tf.argmax(
                predictions[0]
            )

            loss = predictions[
                :,
                predicted_class,
            ]

        gradients = tape.gradient(
            loss,
            conv_outputs,
        )

        pooled_gradients = tf.reduce_mean(

            gradients,

            axis=(0, 1, 2),

        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(

            pooled_gradients * conv_outputs,

            axis=-1,

        )

        heatmap = tf.maximum(
            heatmap,
            0,
        )

        heatmap /= tf.reduce_max(
            heatmap
        ) + 1e-10

        logger.info(
            "Grad-CAM completed."
        )

        return heatmap.numpy()


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    print("GradCAM module loaded successfully.")