"""
TRIDENT TensorFlow Dataset Builder

Creates TensorFlow datasets from extracted
features for efficient model training.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.utils.config import BATCH_SIZE
from src.utils.logger import logger


class DatasetManager:
    """
    Converts NumPy feature arrays and labels into
    TensorFlow datasets for model training.
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        shuffle_buffer: int = 1000,
    ) -> None:

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if shuffle_buffer <= 0:
            raise ValueError(
                "shuffle_buffer must be greater than zero."
            )

        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

    # =========================================================
    # Create Dataset
    # =========================================================

    def create_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        training: bool = True,
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from NumPy arrays.

        Parameters
        ----------
        features:
            Feature tensors with shape:
            (N, 128, 128, 1)

        labels:
            Integer class labels with shape:
            (N,)

        training:
            Whether to shuffle the dataset.

        Returns
        -------
        tf.data.Dataset
            Batched and prefetched TensorFlow dataset.
        """

        logger.info(
            "Creating TensorFlow dataset..."
        )

        # -----------------------------------------------------
        # Validate inputs
        # -----------------------------------------------------

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        labels = np.asarray(
            labels,
            dtype=np.int32,
        )

        if len(features) == 0:
            raise ValueError(
                "Cannot create dataset from empty features."
            )

        if len(labels) == 0:
            raise ValueError(
                "Cannot create dataset from empty labels."
            )

        if len(features) != len(labels):
            raise ValueError(
                "Number of features and labels must match. "
                f"Got {len(features)} features and "
                f"{len(labels)} labels."
            )

        if not np.all(
            np.isfinite(features)
        ):
            raise ValueError(
                "Features contain NaN or infinite values."
            )

        # -----------------------------------------------------
        # Create TensorFlow dataset
        # -----------------------------------------------------

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                features,
                labels,
            )
        )

        # -----------------------------------------------------
        # Shuffle training data
        # -----------------------------------------------------

        if training:

            buffer_size = max(
                self.shuffle_buffer,
                len(features),
            )

            dataset = dataset.shuffle(
                buffer_size=buffer_size,
                reshuffle_each_iteration=True,
            )

        # -----------------------------------------------------
        # Batch
        # -----------------------------------------------------

        dataset = dataset.batch(
            self.batch_size,
            drop_remainder=False,
        )

        # -----------------------------------------------------
        # Prefetch
        # -----------------------------------------------------

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        logger.info(
            "TensorFlow dataset ready: "
            "%d samples, batch_size=%d, training=%s",
            len(features),
            self.batch_size,
            training,
        )

        return dataset


# =============================================================
# Standalone Test
# =============================================================

if __name__ == "__main__":

    X = np.random.rand(
        100,
        128,
        128,
        1,
    ).astype(np.float32)

    y = np.random.randint(
        0,
        4,
        size=100,
        dtype=np.int32,
    )

    manager = DatasetManager()

    dataset = manager.create_dataset(
        X,
        y,
        training=True,
    )

    print()
    print("========== DATASET TEST ==========")
    print()
    print(dataset)

    for batch_x, batch_y in dataset.take(1):

        print()
        print("Feature Shape:")
        print(batch_x.shape)

        print()
        print("Label Shape:")
        print(batch_y.shape)

        print()
        print("Labels:")
        print(batch_y.numpy())

        break