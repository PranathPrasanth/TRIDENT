"""
TRIDENT Mel Spectrogram Generator

Converts underwater acoustic recordings into
Mel Spectrograms for deep learning.
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from utils.config import SAMPLE_RATE
from utils.logger import logger


class MelSpectrogramExtractor:
    """
    Generates Mel Spectrograms from audio waveforms.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ) -> None:

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    # ---------------------------------------------------------

    def extract(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a Mel Spectrogram.
        """

        logger.info(
            "Generating Mel Spectrogram..."
        )

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max,
        )

        logger.info(
            "Mel Spectrogram generated."
        )

        return mel_db

    # ---------------------------------------------------------

    def save_numpy(
        self,
        mel: np.ndarray,
        output_path: str | Path,
    ) -> None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(output_path, mel)

        logger.info(
            f"Saved NumPy feature -> {output_path}"
        )

    # ---------------------------------------------------------

    def save_image(
        self,
        mel: np.ndarray,
        output_path: str | Path,
    ) -> None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(figsize=(10, 4))

        librosa.display.specshow(
            mel,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="mel",
        )

        plt.colorbar(
            format="%+2.0f dB"
        )

        plt.title("Mel Spectrogram")

        plt.tight_layout()

        plt.savefig(output_path)

        plt.close()

        logger.info(
            f"Saved spectrogram image -> {output_path}"
        )


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    from preprocessing.audio_loader import AudioLoader
    from preprocessing.audio_cleaner import AudioCleaner

    loader = AudioLoader()

    cleaner = AudioCleaner()

    extractor = MelSpectrogramExtractor()

    waveform, _ = loader.load_audio(
        "data/raw/example.wav"
    )

    waveform = cleaner.clean(
        waveform
    )

    mel = extractor.extract(
        waveform
    )

    extractor.save_numpy(
        mel,
        "data/features/example.npy",
    )

    extractor.save_image(
        mel,
        "data/spectrograms/example.png",
    )

    print()

    print("Shape")

    print(mel.shape)