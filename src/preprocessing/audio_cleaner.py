"""
TRIDENT Audio Cleaner

Performs preprocessing operations on underwater
acoustic waveforms before feature extraction.
"""

from typing import Optional

import numpy as np

from utils.logger import logger


class AudioCleaner:
    """
    Cleans underwater acoustic recordings.

    Operations
    ----------
    1. Remove DC Offset
    2. Normalize Amplitude
    3. Clip Outliers
    """

    def __init__(
        self,
        clip_threshold: float = 1.0,
    ) -> None:

        self.clip_threshold = clip_threshold

    # ---------------------------------------------------------

    @staticmethod
    def validate_waveform(
        waveform: np.ndarray,
    ) -> None:
        """
        Validate waveform.
        """

        if waveform is None:
            raise ValueError(
                "Waveform is None."
            )

        if len(waveform) == 0:
            raise ValueError(
                "Waveform is empty."
            )

    # ---------------------------------------------------------

    @staticmethod
    def remove_dc_offset(
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Remove DC offset.
        """

        return waveform - np.mean(waveform)

    # ---------------------------------------------------------

    @staticmethod
    def normalize_amplitude(
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Scale waveform between -1 and 1.
        """

        maximum = np.max(
            np.abs(waveform)
        )

        if maximum == 0:
            return waveform

        return waveform / maximum

    # ---------------------------------------------------------

    def clip_outliers(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Remove abnormal spikes.
        """

        return np.clip(
            waveform,
            -self.clip_threshold,
            self.clip_threshold,
        )

    # ---------------------------------------------------------

    @staticmethod
    def rms_energy(
        waveform: np.ndarray,
    ) -> float:
        """
        Compute RMS energy.
        """

        return float(
            np.sqrt(
                np.mean(
                    waveform ** 2
                )
            )
        )

    # ---------------------------------------------------------

    def clean(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Complete cleaning pipeline.
        """

        self.validate_waveform(
            waveform
        )

        logger.info(
            "Cleaning waveform..."
        )

        waveform = self.remove_dc_offset(
            waveform
        )

        waveform = self.normalize_amplitude(
            waveform
        )

        waveform = self.clip_outliers(
            waveform
        )

        logger.info(
            "Cleaning completed."
        )

        return waveform


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    sample = np.array(
        [
            2,
            4,
            6,
            8,
            100
        ],
        dtype=float,
    )

    cleaner = AudioCleaner()

    cleaned = cleaner.clean(sample)

    print()

    print("========== CLEANED ==========")

    print(cleaned)

    print()

    print(
        f"RMS Energy : "
        f"{cleaner.rms_energy(cleaned):.4f}"
    )