"""
TRIDENT Logging Utility

Provides a centralized logger for the entire TRIDENT application.
"""

import logging
from pathlib import Path

from src.utils.config import LOG_DIR


# ---------------------------------------------------------
# Ensure log directory exists
# ---------------------------------------------------------

Path(LOG_DIR).mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Logger Configuration
# ---------------------------------------------------------

logger = logging.getLogger("TRIDENT")

logger.setLevel(logging.INFO)


# Prevent duplicate handlers if the module is imported
# multiple times.
if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    # File handler
    file_handler = logging.FileHandler(
        Path(LOG_DIR) / "trident.log",
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )


logger.propagate = False