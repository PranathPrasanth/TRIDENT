"""
TRIDENT Predictor

Performs inference using the pretrained Marine1 model.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.inference.model_loader import Marine1ModelLoader
from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.audio_cleaner import AudioCleaner
from src.feature_extraction.mel_spectrogram import MelSpectrogramExtractor
from src.utils.logger import logger


class Predictor:
    """
    Performs underwater acoustic classification using Marine1.
    """

    TARGET_SAMPLE_RATE = 16000
    TARGET_DURATION = 10
    TARGET_SAMPLES = (
        TARGET_SAMPLE_RATE * TARGET_DURATION
    )

    def __init__(
        self,
        model_path: str | Path = (
            Path("models")
            / "marine1"
            / "best_model_finetuned.safetensors"
        ),
    ) -> None:

        logger.info("Initializing Marine1 predictor...")

        self.model_loader = Marine1ModelLoader(
            model_path=model_path
        )

        self.model = self.model_loader.get_model()

        self.class_names = (
            self.model_loader.get_classes()
        )

        self.device = torch.device("cpu")

        self.loader = AudioLoader(
            sample_rate=self.TARGET_SAMPLE_RATE,
            mono=True,
        )

        self.cleaner = AudioCleaner()

        self.extractor = MelSpectrogramExtractor(
            sample_rate=self.TARGET_SAMPLE_RATE,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )

        logger.info(
            "Marine1 predictor ready."
        )

    # ---------------------------------------------------------
    # Prepare Waveform
    # ---------------------------------------------------------

    def prepare_waveform(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Convert an arbitrary-length waveform into
        the 10-second input expected by Marine1.

        Longer recordings are truncated.
        Shorter recordings are zero-padded.
        """

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if len(waveform) > self.TARGET_SAMPLES:

            waveform = waveform[
                :self.TARGET_SAMPLES
            ]

        elif len(waveform) < self.TARGET_SAMPLES:

            padding = (
                self.TARGET_SAMPLES
                - len(waveform)
            )

            waveform = np.pad(
                waveform,
                (0, padding),
                mode="constant",
            )

        return waveform.astype(
            np.float32
        )

    # ---------------------------------------------------------
    # Prepare Mel Spectrogram
    # ---------------------------------------------------------

    def prepare_audio(
        self,
        audio_path: str | Path,
    ) -> tuple[torch.Tensor, int]:
        """
        Load and preprocess an audio recording
        into the tensor expected by Marine1.
        """

        waveform, sample_rate = (
            self.loader.load_audio(
                audio_path
            )
        )

        waveform = self.cleaner.clean(
            waveform
        )

        waveform = self.prepare_waveform(
            waveform
        )

        mel = self.extractor.extract(
            waveform
        )

        # Marine1 expects a single-channel
        # log-Mel spectrogram.
        mel = np.asarray(
            mel,
            dtype=np.float32,
        )

        model_input = torch.from_numpy(
            mel
        )

        # [128, time] -> [1, 1, 128, time]
        model_input = model_input.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        model_input = model_input.to(
            self.device
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
        Predict the acoustic source category.
        """

        model_input, _ = (
            self.prepare_audio(
                audio_path
            )
        )

        with torch.no_grad():

            logits = self.model(
                model_input
            )

            probabilities = F.softmax(
                logits,
                dim=1,
            )

        class_index = int(
            torch.argmax(
                probabilities,
                dim=1,
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                class_index,
            ].item()
        )

        predicted_class = (
            self.class_names[
                class_index
            ]
        )

        logger.info(
            f"Prediction completed: "
            f"{predicted_class} "
            f"({confidence:.2%})"
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
        Run inference and return detailed
        information required by the TRIDENT API.
        """

        model_input, sample_rate = (
            self.prepare_audio(
                audio_path
            )
        )

        with torch.no_grad():

            logits = self.model(
                model_input
            )

            probabilities_tensor = (
                F.softmax(
                    logits,
                    dim=1,
                )
            )

        class_index = int(
            torch.argmax(
                probabilities_tensor,
                dim=1,
            ).item()
        )

        confidence = float(
            probabilities_tensor[
                0,
                class_index,
            ].item()
        )

        predicted_class = (
            self.class_names[
                class_index
            ]
        )

        probabilities = {
            class_name: float(
                probabilities_tensor[
                    0,
                    index,
                ].item()
            )
            for index, class_name
            in enumerate(
                self.class_names
            )
        }

        logger.info(
            "Detailed Marine1 prediction completed."
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

    prediction, confidence = (
        predictor.predict(
            "data/raw/example.wav"
        )
    )

    print()
    print(
        "========== MARINE1 PREDICTION =========="
    )
    print(
        f"Target      : {prediction}"
    )
    print(
        f"Confidence  : {confidence:.2%}"
    )