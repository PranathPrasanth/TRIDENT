"""
TRIDENT TensorFlow Dataset Builder

Creates TensorFlow datasets from extracted
features for efficient model training.
"""

import tensorflow as tf
import numpy as np

from utils.logger import logger


class DatasetManager:
    """
    Converts NumPy arrays into TensorFlow datasets.
    """

    def __init__(
        self,
        batch_size: int = 32,
        shuffle_buffer: int = 1000,
    ) -> None:

        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

    # ---------------------------------------------------------

    def create_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        training: bool = True,
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset.
        """

        logger.info(
            "Creating TensorFlow dataset..."
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (features, labels)
        )

        if training:

            dataset = dataset.shuffle(
                self.shuffle_buffer
            )

        dataset = dataset.batch(
            self.batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        logger.info(
            "Dataset ready."
        )

        return dataset


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    X = np.random.rand(
        100,
        128,
        128,
        1,
    )

    y = np.random.randint(
        0,
        4,
        100,
    )

    manager = DatasetManager()

    dataset = manager.create_dataset(
        X,
        y,
    )

    print()

    print("========== DATASET ==========")

    print(dataset)

    print()

    for batch_x, batch_y in dataset.take(1):

        print("Feature Shape")

        print(batch_x.shape)

        print()

        print("Labels")

        print(batch_y)