"""
TRIDENT CNN Model

Defines the Convolutional Neural Network used for
underwater acoustic target classification.
"""

import tensorflow as tf

from utils.logger import logger
from utils.config import INPUT_SHAPE

class TRIDENTModel:
    """
    CNN model for underwater acoustic classification.
    """
    def __init__(
    self,
    input_shape=INPUT_SHAPE,
    num_classes=4,
    )-> None:

        self.input_shape = input_shape
        self.num_classes = num_classes

    # ---------------------------------------------------------

    def build(self) -> tf.keras.Model:
        """
        Build and compile the CNN.
        """

        logger.info("Building CNN model...")

        model = tf.keras.Sequential(

            [

                tf.keras.layers.Input(
                    shape=self.input_shape
                ),

                # -------------------------

                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="same",
                ),

                tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2)
                ),

                # -------------------------

                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="same",
                ),

                tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2)
                ),

                # -------------------------

                tf.keras.layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    activation="relu",
                    padding="same",
                ),

                tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2)
                ),

                # -------------------------

                tf.keras.layers.Flatten(),

                tf.keras.layers.Dense(
                    256,
                    activation="relu",
                ),

                tf.keras.layers.Dropout(
                    0.5
                ),

                tf.keras.layers.Dense(
                    self.num_classes,
                    activation="softmax",
                ),

            ]

        )

        model.compile(

            optimizer="adam",

            loss="sparse_categorical_crossentropy",

            metrics=["accuracy"],

        )

        logger.info("CNN model built successfully.")

        return model


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    cnn = TRIDENTModel()

    model = cnn.build()

    model.summary()