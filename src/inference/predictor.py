"""
TRIDENT Predictor

Performs inference using a trained CNN model.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing.audio_loader import AudioLoader
from preprocessing.audio_cleaner import AudioCleaner
from feature_extraction.mel_spectrogram import MelSpectrogramExtractor
from feature_extraction.normalizer import FeatureNormalizer

from utils.config import MODEL_DIR
from utils.logger import logger


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

        logger.info("Predictor ready.")

    # ---------------------------------------------------------

    def predict(
        self,
        audio_path: str | Path,
        class_names: list[str],
    ) -> tuple[str, float]:
        """
        Predict the class of an audio recording.
        """

        waveform, _ = self.loader.load_audio(
            audio_path
        )

        waveform = self.cleaner.clean(
            waveform
        )

        mel = self.extractor.extract(
            waveform
        )

        mel = self.normalizer.normalize(
            mel
        )

        mel = np.expand_dims(
            mel,
            axis=-1,
        )

        mel = np.expand_dims(
            mel,
            axis=0,
        )

        prediction = self.model.predict(
            mel,
            verbose=0,
        )

        class_index = np.argmax(
            prediction
        )

        confidence = float(
            prediction[0][class_index]
        )

        logger.info("Prediction completed.")

        return (
            class_names[class_index],
            confidence,
        )


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    predictor = Predictor()

    classes = [
        "Submarine",
        "Ship",
        "Torpedo",
        "Background",
    ]

    prediction, confidence = predictor.predict(
        "data/raw/example.wav",
        classes,
    )

    print()

    print("========== PREDICTION ==========")

    print(f"Class      : {prediction}")

    print(f"Confidence : {confidence:.2%}")