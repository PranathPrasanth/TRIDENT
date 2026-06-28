"""
TRIDENT Feature Extraction Package

This package converts cleaned underwater acoustic recordings
into machine learning features.

Modules
-------
mel_spectrogram
    Generates Mel Spectrograms.

mfcc
    Extracts Mel-Frequency Cepstral Coefficients.

normalizer
    Normalizes extracted features.

visualization
    Visualizes feature representations.
"""

__all__ = [
    "mel_spectrogram",
    "mfcc",
    "normalizer",
    "visualization",
]