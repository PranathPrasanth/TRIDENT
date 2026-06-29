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