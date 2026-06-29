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

    def _find_last_conv_layer(
        self,
    ) -> str:
        """
        Find the last Conv2D layer automatically.
        """

        for layer in reversed(
            self.model.layers
        ):

            if isinstance(
                layer,
                tf.keras.layers.Conv2D,
            ):

                return layer.name

        raise ValueError(
            "No Conv2D layer found in the model."
        )

    # ---------------------------------------------------------

    def generate(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.
        """

        logger.info(
            "Generating Grad-CAM heatmap..."
        )

        last_conv_layer_name = (
            self._find_last_conv_layer()
        )

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
                predictions[0],
                output_type=tf.int32,
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

        conv_outputs = tf.squeeze(
            conv_outputs,
            axis=0,
        )

        heatmap = tf.reduce_sum(

            pooled_gradients * conv_outputs,

            axis=-1,

        )

        heatmap = tf.maximum(
            heatmap,
            0,
        )

        heatmap /= (
            tf.reduce_max(heatmap)
            + 1e-10
        )

        logger.info(
            "Grad-CAM completed."
        )

        return heatmap.numpy()