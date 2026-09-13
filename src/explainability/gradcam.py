"""
TRIDENT Grad-CAM Generator

Generates Grad-CAM heatmaps for explaining
CNN predictions.
"""

import numpy as np
import tensorflow as tf

from src.utils.logger import logger


class GradCAM:
    """
    Generates Grad-CAM heatmaps.
    """

    def __init__(
        self,
        model: tf.keras.Model,
    ) -> None:

        self.model = model

        # Find the final convolutional layer once.
        self.last_conv_layer = None

        for layer in reversed(self.model.layers):
            if isinstance(
                layer,
                tf.keras.layers.Conv2D,
            ):
                self.last_conv_layer = layer
                break

        if self.last_conv_layer is None:
            raise ValueError(
                "No Conv2D layer found in the model."
            )

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

        image = tf.convert_to_tensor(
            image,
            dtype=tf.float32,
        )

        with tf.GradientTape() as tape:

            # Watch the activation of the final
            # convolutional layer.
            conv_outputs = self.last_conv_layer(
                image,
            )

            # Continue the model forward pass from
            # the convolutional layer to the output.
            x = conv_outputs

            layer_index = self.model.layers.index(
                self.last_conv_layer
            )

            for layer in self.model.layers[
                layer_index + 1:
            ]:
                x = layer(x)

            predictions = x

            predicted_class = tf.argmax(
                predictions[0],
                output_type=tf.int32,
            )

            loss = predictions[
                0,
                predicted_class,
            ]

        gradients = tape.gradient(
            loss,
            conv_outputs,
        )

        if gradients is None:
            raise ValueError(
                "Unable to compute Grad-CAM gradients."
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

        maximum = tf.reduce_max(heatmap)

        heatmap = heatmap / (
            maximum + 1e-10
        )

        logger.info(
            "Grad-CAM completed."
        )

        return heatmap.numpy()