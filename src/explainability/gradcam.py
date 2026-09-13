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
    Generates Grad-CAM heatmaps for a CNN model.
    """

    def __init__(
        self,
        model: tf.keras.Model,
    ) -> None:

        self.model = model

        self.last_conv_layer = self._find_last_conv_layer()

        logger.info(
            "Grad-CAM target layer: %s",
            self.last_conv_layer.name,
        )

    def _find_last_conv_layer(
        self,
    ) -> tf.keras.layers.Conv2D:
        """
        Find the final Conv2D layer automatically.
        """

        for layer in reversed(self.model.layers):

            if isinstance(
                layer,
                tf.keras.layers.Conv2D,
            ):
                return layer

        raise ValueError(
            "No Conv2D layer found in the model."
        )

    def generate(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Parameters
        ----------
        image:
            Model input with shape:
            (1, 128, 128, 1)

        Returns
        -------
        np.ndarray
            Normalized heatmap.
        """

        logger.info(
            "Generating Grad-CAM heatmap..."
        )

        image = tf.convert_to_tensor(
            image,
            dtype=tf.float32,
        )

        with tf.GradientTape() as tape:

            # Build a model that returns:
            # 1. Final convolutional activations
            # 2. Final classification prediction
            grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[
                    self.last_conv_layer.output,
                    self.model.layers[-1].output,
                ],
            )

            conv_outputs, predictions = grad_model(
                image,
                training=False,
            )

            predicted_class = tf.argmax(
                predictions[0],
                output_type=tf.int32,
            )

            class_score = predictions[
                0,
                predicted_class,
            ]

        gradients = tape.gradient(
            class_score,
            conv_outputs,
        )

        if gradients is None:
            raise ValueError(
                "Unable to compute Grad-CAM gradients."
            )

        # Global average pooling of gradients.
        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2),
        )

        # Remove batch dimension.
        conv_outputs = conv_outputs[0]

        # Weight each feature map by its
        # corresponding gradient importance.
        heatmap = tf.reduce_sum(
            pooled_gradients * conv_outputs,
            axis=-1,
        )

        # Keep only positive activations.
        heatmap = tf.maximum(
            heatmap,
            0,
        )

        # Normalize to [0, 1].
        maximum = tf.reduce_max(heatmap)

        heatmap = heatmap / (
            maximum + 1e-10
        )

        logger.info(
            "Grad-CAM completed."
        )

        return heatmap.numpy()