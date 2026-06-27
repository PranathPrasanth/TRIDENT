"""
TRIDENT Audio Loader

Loads underwater acoustic recordings into memory and converts them
into a standardized waveform suitable for the preprocessing pipeline.
"""

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np

from utils.config import SAMPLE_RATE, MONO
from utils.logger import logger
from utils.exceptions import AudioLoadError


class AudioLoader:
    """
    Loads underwater acoustic recordings.

    Features
    --------
    - File validation
    - Automatic resampling
    - Mono conversion
    - Error handling
    - Logging
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        mono: bool = MONO,
    ) -> None:
        """
        Initialize the audio loader.

        Parameters
        ----------
        sample_rate : int
            Desired sampling rate.

        mono : bool
            Whether to convert stereo recordings to mono.
        """

        self.sample_rate = sample_rate
        self.mono = mono

    # ---------------------------------------------------------

    def validate_file(self, file_path: Path) -> None:
        """
        Validate the input audio file.

        Parameters
        ----------
        file_path : Path
            Path to audio file.

        Raises
        ------
        AudioLoadError
            If the file does not exist or is invalid.
        """

        if not file_path.exists():
            raise AudioLoadError(
                f"Audio file not found: {file_path}"
            )

        if file_path.suffix.lower() != ".wav":
            raise AudioLoadError(
                f"Unsupported file type: {file_path.suffix}"
            )

    # ---------------------------------------------------------

    def load_audio(
        self,
        file_path: str | Path
    ) -> Tuple[np.ndarray, int]:
        """
        Load an underwater audio recording.

        Parameters
        ----------
        file_path : str | Path

        Returns
        -------
        tuple

        waveform : np.ndarray

        sample_rate : int
        """

        file_path = Path(file_path)

        self.validate_file(file_path)

        logger.info(
            f"Loading audio: {file_path.name}"
        )

        try:

            waveform, sample_rate = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=self.mono,
            )

        except Exception as error:

            logger.error(error)

            raise AudioLoadError(
                f"Unable to load {file_path.name}"
            ) from error

        logger.info(
            f"Loaded {file_path.name} "
            f"({len(waveform)} samples)"
        )

        return waveform, sample_rate

    # ---------------------------------------------------------

    @staticmethod
    def duration(
        waveform: np.ndarray,
        sample_rate: int,
    ) -> float:
        """
        Calculate audio duration.

        Returns
        -------
        float
            Duration in seconds.
        """

        return len(waveform) / sample_rate


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    loader = AudioLoader()

    try:

        waveform, sr = loader.load_audio(
            "data/raw/example.wav"
        )

        print()

        print("========== AUDIO INFO ==========")

        print(f"Sample Rate : {sr}")

        print(
            f"Duration : "
            f"{loader.duration(waveform, sr):.2f} sec"
        )

        print(f"Samples : {len(waveform)}")

        print()

        print("First 10 Samples")

        print(waveform[:10])

    except AudioLoadError as error:

        logger.error(error)