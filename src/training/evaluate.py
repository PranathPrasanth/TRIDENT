"""
TRIDENT Model Evaluation

Evaluates the trained CNN model.
"""

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils.logger import logger


class ModelEvaluator:
    """
    Evaluates a trained CNN model.
    """

    def evaluate(
        self,
        model: tf.keras.Model,
        test_dataset: tf.data.Dataset,
    ) -> dict:
        """
        Evaluate the trained model.
        """

        logger.info("Evaluating model...")

        y_true = []
        y_pred = []

        for features, labels in test_dataset:

            predictions = model.predict(
                features,
                verbose=0,
            )

            predicted_labels = np.argmax(
                predictions,
                axis=1,
            )

            y_true.extend(labels.numpy())

            y_pred.extend(predicted_labels)

        results = {

            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),

            "precision": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "recall": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "f1_score": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
            ),

            "classification_report": classification_report(
                y_true,
                y_pred,
                zero_division=0,
                target_names=class_names,
            ),

        }

        logger.info("Evaluation completed.")

        return results


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    print("ModelEvaluator module loaded successfully.")