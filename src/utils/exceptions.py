"""
TRIDENT Custom Exceptions

Centralized exception definitions for the TRIDENT platform.
"""


class TRIDENTError(Exception):
    """Base exception for all TRIDENT-specific errors."""

    pass


class AudioLoadError(TRIDENTError):
    """Raised when an audio file cannot be loaded or validated."""

    pass


class DatasetError(TRIDENTError):
    """Raised when dataset construction or validation fails."""

    pass


class FeatureExtractionError(TRIDENTError):
    """Raised when feature extraction fails."""

    pass


class ModelError(TRIDENTError):
    """Raised when model construction, training, or loading fails."""

    pass


class InferenceError(TRIDENTError):
    """Raised when model inference fails."""

    pass