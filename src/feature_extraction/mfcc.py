"""
TRIDENT MFCC Feature Extractor

Extracts Mel-Frequency Cepstral Coefficients
from underwater acoustic recordings.
"""

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from utils.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MFCC,
)

from utils.logger import logger


class MFCCExtractor:
    """
    Extract MFCC features from audio.
    """

    def __init__(
        self,
        sample_rate=SAMPLE_RATE
        n_fft=N_FFT
        hop_length=HOP_LENGTH
        n_mfcc=N_MFCC,
    ) -> None:

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    # ---------------------------------------------------------

    def extract(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Extract MFCC features.
        """

        logger.info("Extracting MFCC...")

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        logger.info("MFCC extraction completed.")

        return mfcc

    # ---------------------------------------------------------

    def save_numpy(
        self,
        mfcc: np.ndarray,
        output_path: str | Path,
    ) -> None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(output_path, mfcc)

        logger.info(
            f"Saved MFCC -> {output_path}"
        )

    # ---------------------------------------------------------

    def save_image(
        self,
        mfcc: np.ndarray,
        output_path: str | Path,
    ) -> None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(figsize=(10, 4))

        plt.imshow(
            mfcc,
            origin="lower",
            aspect="auto",
        )

        plt.colorbar()

        plt.title("MFCC")

        plt.xlabel("Frames")

        plt.ylabel("Coefficients")

        plt.tight_layout()

        plt.savefig(output_path)

        plt.close()

        logger.info(
            f"Saved MFCC image -> {output_path}"
        )


# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    from preprocessing.audio_loader import AudioLoader
    from preprocessing.audio_cleaner import AudioCleaner

    loader = AudioLoader()

    cleaner = AudioCleaner()

    extractor = MFCCExtractor()

    waveform, _ = loader.load_audio(
        "data/raw/example.wav"
    )

    waveform = cleaner.clean(waveform)

    mfcc = extractor.extract(
        waveform
    )

    extractor.save_numpy(
        mfcc,
        "data/features/example_mfcc.npy",
    )

    extractor.save_image(
        mfcc,
        "data/features/example_mfcc.png",
    )

    print()

    print("MFCC Shape")

    print(mfcc.shape)