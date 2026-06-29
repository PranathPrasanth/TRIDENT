"""
TRIDENT Mel Spectrogram Generator

Converts underwater acoustic recordings into
Mel Spectrograms for deep learning.
"""

import librosa
import numpy as np

from utils.config import SAMPLE_RATE
from utils.logger import logger
from utils.helpers import save_numpy

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

    save_numpy(
        mel,
        "data/features/example.npy",
    )

    print()

    print("========== MEL SPECTROGRAM ==========")

    print(f"Shape : {mel.shape}")

    print(f"Data Type : {mel.dtype}")

    print(f"Minimum Value : {mel.min():.2f}")

    print(f"Maximum Value : {mel.max():.2f}")