"""
TRIDENT Predictor

Performs inference using a trained CNN model.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.audio_cleaner import AudioCleaner
from src.feature_extraction.mel_spectrogram import MelSpectrogramExtractor
from src.feature_extraction.normalizer import FeatureNormalizer

from src.utils.config import (
    MODEL_DIR,
    METADATA_DIR,
)
from src.utils.logger import logger


class Predictor:
    """
    Performs prediction on new audio recordings.
    """

    def __init__(
        self,
        model_path: str | Path = Path(MODEL_DIR) / "best_model.keras",
    ) -> None:

        logger.info("Loading trained model...")

        self.model = tf.keras.models.load_model(
            model_path
        )

        self.loader = AudioLoader()

        self.cleaner = AudioCleaner()

        self.extractor = MelSpectrogramExtractor()

        self.normalizer = FeatureNormalizer()

        with open(
            METADATA_DIR / "class_mapping.json",
            "r",
            encoding="utf-8",
        ) as file:

            self.class_mapping = json.load(file)

        logger.info("Predictor ready.")

    # ---------------------------------------------------------
    # Resize Mel Spectrogram
    # ---------------------------------------------------------

    def resize_mel(
        self,
        mel: np.ndarray,
        target_height: int = 128,
        target_width: int = 128,
    ) -> np.ndarray:
        """
        Convert a Mel spectrogram into the same fixed size
        used during model training.
        """

        height, width = mel.shape

        if height > target_height:

            mel = mel[
                :target_height,
                :
            ]

        elif height < target_height:

            padding = target_height - height

            mel = np.pad(
                mel,
                (
                    (0, padding),
                    (0, 0),
                ),
                mode="constant",
            )

        if width > target_width:

            mel = mel[
                :,
                :target_width,
            ]

        elif width < target_width:

            padding = target_width - width

            mel = np.pad(
                mel,
                (
                    (0, 0),
                    (0, padding),
                ),
                mode="constant",
            )

        return mel.astype(
            np.float32
        )

    # ---------------------------------------------------------
    # Prepare Audio
    # ---------------------------------------------------------

    def prepare_audio(
        self,
        audio_path: str | Path,
    ) -> tuple[np.ndarray, int]:
        """
        Load and preprocess an audio file into the exact
        tensor representation expected by the CNN.
        """

        waveform, sample_rate = (
            self.loader.load_audio(
                audio_path
            )
        )

        waveform = self.cleaner.clean(
            waveform
        )

        mel = self.extractor.extract(
            waveform
        )

        # IMPORTANT:
        # Same normalization used during training.
        mel = self.normalizer.normalize(
            mel
        )

        # IMPORTANT:
        # Same fixed dimensions used during training.
        mel = self.resize_mel(
            mel
        )

        mel = np.expand_dims(
            mel,
            axis=-1,
        )

        model_input = np.expand_dims(
            mel,
            axis=0,
        )

        return (
            model_input,
            sample_rate,
        )

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    def predict(
        self,
        audio_path: str | Path,
    ) -> tuple[str, float]:
        """
        Predict the class of an audio recording.
        """

        model_input, _ = (
            self.prepare_audio(
                audio_path
            )
        )

        prediction = self.model.predict(
            model_input,
            verbose=0,
        )

        class_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[0][class_index]
        )

        predicted_class = self.class_mapping[
            str(class_index)
        ]

        logger.info(
            "Prediction completed."
        )

        return (
            predicted_class,
            confidence,
        )

    # ---------------------------------------------------------
    # Detailed Prediction
    # ---------------------------------------------------------

    def predict_with_details(
        self,
        audio_path: str | Path,
    ) -> dict:
        """
        Run inference and return information required
        by the TRIDENT API.
        """

        model_input, sample_rate = (
            self.prepare_audio(
                audio_path
            )
        )

        prediction = self.model.predict(
            model_input,
            verbose=0,
        )

        class_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[0][class_index]
        )

        predicted_class = self.class_mapping[
            str(class_index)
        ]

        probabilities = {
            self.class_mapping[str(index)]: float(
                probability
            )
            for index, probability
            in enumerate(prediction[0])
        }

        logger.info(
            "Detailed prediction completed."
        )

        return {
            "class": predicted_class,
            "confidence": confidence,
            "class_index": class_index,
            "probabilities": probabilities,
            "sample_rate": sample_rate,
            "input_shape": list(
                model_input.shape
            ),
        }


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    predictor = Predictor()

    prediction, confidence = predictor.predict(
        "data/raw/example.wav"
    )

    print()

    print(
        "========== PREDICTION =========="
    )

    print(
        f"Target      : {prediction}"
    )

    print(
        f"Confidence  : {confidence:.2%}"
    )