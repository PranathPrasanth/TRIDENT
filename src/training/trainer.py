"""
TRIDENT Model Trainer

Handles training of the CNN model.
"""

from pathlib import Path

import tensorflow as tf

from utils.logger import logger
from utils.helpers import ensure_directory
from utils.config import (
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

    def train(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
    ) -> tf.keras.callbacks.History:
        """
        Train the CNN model.
        """

        logger.info("Training started...")

        checkpoint = tf.keras.callbacks.ModelCheckpoint(

            filepath=self.model_dir / "best_model.keras",

            monitor=CHECKPOINT_MONITOR,

            save_best_only=True,

            verbose=1,

        )

        early_stop = tf.keras.callbacks.EarlyStopping(

            monitor=EARLY_STOP_MONITOR,

            patience=PATIENCE,

            restore_best_weights=True,

            verbose=1,

        )

        history = model.fit(

            train_dataset,

            validation_data=validation_dataset,

            epochs=self.epochs,

            callbacks=[
                checkpoint,
                early_stop,
            ],

        )

        logger.info("Training completed.")

        return history


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    print("ModelTrainer module loaded successfully.")