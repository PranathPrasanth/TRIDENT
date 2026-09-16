"""
TRIDENT Model Trainer

Handles training of the CNN model with class-weighted learning.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.utils.logger import logger
from src.utils.helpers import ensure_directory
from src.utils.config import (
    EPOCHS,
    MODEL_DIR,
    PATIENCE,
    CHECKPOINT_MONITOR,
    EARLY_STOP_MONITOR,
)


class ModelTrainer:
    """
    Handles CNN model training.
    """

    def __init__(
        self,
        epochs: int = EPOCHS,
        model_dir: str = MODEL_DIR,
    ) -> None:

        self.epochs = epochs
        self.model_dir = Path(model_dir)

        ensure_directory(self.model_dir)

    # ---------------------------------------------------------
    # Class Weight Calculation
    # ---------------------------------------------------------

    def calculate_class_weights(
        self,
        labels: np.ndarray,
    ) -> dict[int, float]:
        """
        Calculate balanced class weights from training labels.

        Rare classes receive higher weights so that the model
        does not simply favor the majority class.
        """

        labels = np.asarray(labels)

        classes, counts = np.unique(
            labels,
            return_counts=True,
        )

        total_samples = len(labels)
        number_of_classes = len(classes)

        class_weights = {}

        for class_index, count in zip(
            classes,
            counts,
        ):

            weight = (
                total_samples
                / (
                    number_of_classes
                    * count
                )
            )

            class_weights[int(class_index)] = float(
                weight
            )

        logger.info(
            "Calculated class weights: %s",
            class_weights,
        )

        for class_index, count in zip(
            classes,
            counts,
        ):

            logger.info(
                "Class %d | samples=%d | weight=%.4f",
                class_index,
                count,
                class_weights[int(class_index)],
            )

        return class_weights

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        class_weights: dict[int, float] | None = None,
    ) -> tf.keras.callbacks.History:
        """
        Train the CNN model.

        Parameters
        ----------
        model:
            Compiled TensorFlow model.

        train_dataset:
            Training TensorFlow dataset.

        validation_dataset:
            Validation TensorFlow dataset.

        class_weights:
            Optional class-weight mapping used to compensate
            for class imbalance.
        """

        logger.info(
            "Training started..."
        )

        if class_weights is not None:

            logger.info(
                "Class-weighted training enabled."
            )

        checkpoint = (
            tf.keras.callbacks.ModelCheckpoint(

                filepath=(
                    self.model_dir
                    / "best_model.keras"
                ),

                monitor=CHECKPOINT_MONITOR,

                save_best_only=True,

                verbose=1,

            )
        )

        early_stop = (
            tf.keras.callbacks.EarlyStopping(

                monitor=EARLY_STOP_MONITOR,

                patience=PATIENCE,

                restore_best_weights=True,

                verbose=1,

            )
        )

        history = model.fit(

            train_dataset,

            validation_data=validation_dataset,

            epochs=self.epochs,

            callbacks=[
                checkpoint,
                early_stop,
            ],

            class_weight=class_weights,

        )

        logger.info(
            "Training completed."
        )

        return history


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    print(
        "ModelTrainer module loaded successfully."
    )