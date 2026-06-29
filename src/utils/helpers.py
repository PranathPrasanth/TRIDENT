"""
TRIDENT Helper Utilities

Common helper functions used across the project.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.logger import logger


def ensure_directory(directory: str | Path) -> Path:
    """
    Create a directory if it does not exist.
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def save_numpy(
    array: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save a NumPy array.
    """

    output_path = Path(output_path)

    ensure_directory(output_path.parent)

    np.save(output_path, array)

    logger.info(
        f"Saved NumPy file -> {output_path}"
    )


def save_figure(
    output_path: str | Path,
) -> None:
    """
    Save the current matplotlib figure.
    """

    output_path = Path(output_path)

    ensure_directory(output_path.parent)

    plt.savefig(output_path)

    logger.info(
        f"Saved figure -> {output_path}"
    )

    plt.close()