"""
TRIDENT Visualization Module

Provides visualization utilities for
waveforms, Mel Spectrograms and MFCCs.
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from utils.config import SAMPLE_RATE
from utils.logger import logger


class FeatureVisualizer:
    """
    Visualize audio features.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:

        self.sample_rate = sample_rate

    # ---------------------------------------------------------

    def plot_waveform(
        self,
        waveform: np.ndarray,
        title: str = "Waveform",
    ) -> None:

        plt.figure(figsize=(12, 4))

        librosa.display.waveshow(
            waveform,
            sr=self.sample_rate,
        )

        plt.title(title)

        plt.xlabel("Time (seconds)")

        plt.ylabel("Amplitude")

        plt.tight_layout()

        plt.show()

    # ---------------------------------------------------------

    def plot_mel(
        self,
        mel: np.ndarray,
        title: str = "Mel Spectrogram",
    ) -> None:

        plt.figure(figsize=(10, 4))

        librosa.display.specshow(
            mel,
            sr=self.sample_rate,
            x_axis="time",
            y_axis="mel",
        )

        plt.colorbar(format="%+2.0f dB")

        plt.title(title)

        plt.tight_layout()

        plt.show()

    # ---------------------------------------------------------

    def plot_mfcc(
        self,
        mfcc: np.ndarray,
        title: str = "MFCC",
    ) -> None:

        plt.figure(figsize=(10, 4))

        librosa.display.specshow(
            mfcc,
            x_axis="time",
            sr=self.sample_rate,
        )

        plt.colorbar()

        plt.title(title)

        plt.tight_layout()

        plt.show()

    # ---------------------------------------------------------

    def save_figure(
        self,
        output_path: str | Path,
    ) -> None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(output_path)

        logger.info(
            f"Saved figure -> {output_path}"
        )

        plt.close()


# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    from preprocessing.audio_loader import AudioLoader
    from preprocessing.audio_cleaner import AudioCleaner

    from feature_extraction.mel_spectrogram import (
        MelSpectrogramExtractor,
    )

    from feature_extraction.mfcc import (
        MFCCExtractor,
    )

    loader = AudioLoader()

    cleaner = AudioCleaner()

    mel_extractor = MelSpectrogramExtractor()

    mfcc_extractor = MFCCExtractor()

    visualizer = FeatureVisualizer()

    waveform, _ = loader.load_audio(
        "data/raw/example.wav"
    )

    waveform = cleaner.clean(
        waveform
    )

    mel = mel_extractor.extract(
        waveform
    )

    mfcc = mfcc_extractor.extract(
        waveform
    )

    visualizer.plot_waveform(
        waveform
    )

    visualizer.plot_mel(
        mel
    )

    visualizer.plot_mfcc(
        mfcc
    )