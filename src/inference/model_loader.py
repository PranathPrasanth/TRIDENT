"""
TRIDENT Marine1 Model Loader

Loads the pretrained Marine1 ResNet18 model and its weights.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from safetensors.torch import load_file

from src.utils.logger import logger


MARINE1_CLASSES = [
    "vessel",
    "marine_animal",
    "natural_sound",
    "other_anthropogenic",
]


class Marine1ModelLoader:
    """
    Loads the pretrained Marine1 ResNet18 classifier.
    """

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Marine1 model not found: {self.model_path}"
            )

        self.device = torch.device("cpu")

        logger.info(
            f"Loading Marine1 model from {self.model_path}"
        )

        self.model = self._build_model()

        state_dict = load_file(
            str(self.model_path),
            device="cpu",
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info("Marine1 model loaded successfully.")

    def _build_model(self) -> nn.Module:
        """
        Reconstruct the Marine1 ResNet18 architecture.
        """

        model = models.resnet18(
            weights=None
        )

        # Marine1 operates on single-channel
        # Mel spectrograms.
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Marine1 uses four output classes.
        model.fc = nn.Linear(
            model.fc.in_features,
            len(MARINE1_CLASSES),
        )

        return model

    def get_model(self) -> nn.Module:
        """
        Return the loaded model.
        """

        return self.model

    def get_classes(self) -> list[str]:
        """
        Return Marine1 class names.
        """

        return MARINE1_CLASSES.copy()