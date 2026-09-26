"""
TRIDENT Marine1 Model Loader

Loads the pretrained Marine1 ResNet18 model and its weights.
"""

import ast
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file
from torchvision import models

from src.utils.logger import logger


class Marine1Network(nn.Module):
    """
    Exact Marine1 architecture from the released inference code.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()

        resnet = models.resnet18(
            weights=None
        )

        # Grayscale Mel spectrogram input.
        self.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = False,
    ):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        features = torch.flatten(
            x,
            1,
        )

        logits = self.classifier(
            features
        )

        if return_confidence:
            confidence = self.confidence_head(
                features
            )

            return logits, confidence

        return logits


class Marine1ModelLoader:
    """
    Loads the pretrained Marine1 model.
    """

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Marine1 model not found: "
                f"{self.model_path}"
            )

        self.device = torch.device(
            "cpu"
        )

        logger.info(
            f"Loading Marine1 model from "
            f"{self.model_path}"
        )

        self.class_to_id = (
            self._load_class_mapping()
        )

        self.id_to_class = {
            value: key
            for key, value
            in self.class_to_id.items()
        }

        self.class_names = [
            self.id_to_class[index]
            for index in range(
                len(self.id_to_class)
            )
        ]

        self.model = Marine1Network(
            num_classes=len(
                self.class_names
            )
        )

        state_dict = load_file(
            str(self.model_path),
            device="cpu",
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        logger.info(
            "Marine1 model loaded successfully."
        )

        logger.info(
            f"Marine1 classes: "
            f"{self.class_names}"
        )

    def _load_class_mapping(self) -> dict:
        """
        Read the class mapping stored in the
        Safetensors metadata.
        """

        with safe_open(
            str(self.model_path),
            framework="pt",
            device="cpu",
        ) as file:

            metadata = file.metadata()

        class_mapping = ast.literal_eval(
            metadata.get(
                "class_to_id",
                "{}",
            )
        )

        if not class_mapping:

            class_mapping = {
                "vessel": 0,
                "marine_animal": 1,
                "natural_sound": 2,
                "other_anthropogenic": 3,
            }

        return class_mapping

    def get_model(self) -> nn.Module:
        """
        Return the loaded Marine1 model.
        """

        return self.model

    def get_classes(self) -> list[str]:
        """
        Return Marine1 class names.
        """

        return self.class_names.copy()

    def get_device(self) -> torch.device:
        """
        Return the inference device.
        """

        return self.device