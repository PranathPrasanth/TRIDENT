"""
TRIDENT Model Evaluator

Evaluates the trained CNN model.
"""

import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from utils.config import METADATA_DIR
from utils.logger import logger


class ModelEvaluator:
    """
    Evaluates the performance of
    the trained CNN model.
    """

    def __init__(
        self,
        model: tf.keras.Model,
    ) -> None:

        self.model = model

        with open(
            METADATA_DIR / "class_mapping.json",
            "r",
            encoding="utf-8",
        ) as file:

            self.class_mapping = json.load(file)

        self.class_names = [

            self.class_mapping[str(i)]

            for i in range(
                len(self.class_mapping)
            )

        ]

    # ---------------------------------------------------------

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        Evaluate the trained model.
        """

        logger.info(
            "Evaluating model..."
        )

        loss, accuracy = self.model.evaluate(
            X_test,
            y_test,
            verbose=0,
        )

        predictions = self.model.predict(
            X_test,
            verbose=0,
        )

        predicted_labels = np.argmax(
            predictions,
            axis=1,
        )

        print()

        print("========== MODEL PERFORMANCE ==========")

        print(f"Loss      : {loss:.4f}")

        print(f"Accuracy  : {accuracy:.2%}")

        print()

        print("========== CLASSIFICATION REPORT ==========")

        print(

            classification_report(

                y_test,

                predicted_labels,

                target_names=self.class_names,

            )

        )

        print()

        print("========== CONFUSION MATRIX ==========")

        print(

            confusion_matrix(

                y_test,

                predicted_labels,

            )

        )

        logger.info(
            "Evaluation completed."
        )


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    print()

    print(
        "ModelEvaluator module loaded successfully."
    )