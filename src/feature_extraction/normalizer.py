"""
TRIDENT Feature Normalizer

Provides normalization utilities for machine learning
feature vectors.
"""

from pathlib import Path
import json

import numpy as np

from utils.logger import logger


class FeatureNormalizer:
    """
    Performs feature normalization for ML models.

    Supported Methods
    -----------------
    - Standardization (Z-score)
    - Min-Max Scaling
    - L2 Normalization
    """

    def __init__(self):

        self.mean = None
        self.std = None
        self.minimum = None
        self.maximum = None

    # ---------------------------------------------------------

    @staticmethod
    def validate(features: np.ndarray) -> None:
        """
        Validate feature vector.
        """

        if features is None:
            raise ValueError("Features cannot be None.")

        if len(features) == 0:
            raise ValueError("Empty feature vector.")

    # ---------------------------------------------------------

    def fit(self, features: np.ndarray) -> None:
        """
        Learn normalization statistics.
        """

        self.validate(features)

        self.mean = float(np.mean(features))
        self.std = float(np.std(features))

        self.minimum = float(np.min(features))
        self.maximum = float(np.max(features))

        logger.info("Normalization statistics computed.")

    # ---------------------------------------------------------

    def standardize(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Z-score normalization.
        """

        self.validate(features)

        if self.mean is None:
            self.fit(features)

        if self.std == 0:
            return features

        return (features - self.mean) / self.std

    # ---------------------------------------------------------

    def min_max_scale(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Scale between 0 and 1.
        """

        self.validate(features)

        if self.minimum is None:
            self.fit(features)

        if self.maximum == self.minimum:
            return features

        return (
            (features - self.minimum)
            /
            (self.maximum - self.minimum)
        )

    # ---------------------------------------------------------

    @staticmethod
    def l2_normalize(
        features: np.ndarray,
    ) -> np.ndarray:
        """
        L2 vector normalization.
        """

        norm = np.linalg.norm(features)

        if norm == 0:
            return features

        return features / norm

    # ---------------------------------------------------------

    def save_statistics(
        self,
        output_path: str | Path,
    ) -> None:
        """
        Save normalization statistics.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        statistics = {
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                statistics,
                file,
                indent=4,
            )

        logger.info(
            "Normalization statistics saved."
        )

    # ---------------------------------------------------------

    def load_statistics(
        self,
        input_path: str | Path,
    ) -> None:
        """
        Load previously saved statistics.
        """

        with open(
            input_path,
            "r",
            encoding="utf-8",
        ) as file:

            statistics = json.load(file)

        self.mean = statistics["mean"]
        self.std = statistics["std"]
        self.minimum = statistics["minimum"]
        self.maximum = statistics["maximum"]

        logger.info(
            "Normalization statistics loaded."
        )


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    sample = np.array(
        [10, 20, 30, 40, 50],
        dtype=float,
    )

    normalizer = FeatureNormalizer()

    print()

    print("Standardized")

    print(
        normalizer.standardize(sample)
    )

    print()

    print("Min-Max")

    print(
        normalizer.min_max_scale(sample)
    )

    print()

    print("L2")

    print(
        normalizer.l2_normalize(sample)
    )

    normalizer.save_statistics(
        "data/metadata/normalization.json"
    )