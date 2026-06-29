"""
TRIDENT Dataset Builder

Builds machine learning datasets from underwater
acoustic recordings.
"""
import json

from utils.config import (
    METADATA_DIR,
)

from pathlib import Path
import random
from typing import Dict, List

from sklearn.model_selection import train_test_split

from preprocessing.audio_loader import AudioLoader
from preprocessing.audio_cleaner import AudioCleaner
from utils.config import (
    RAW_DATA_DIR,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)
from utils.logger import logger
from utils.exceptions import DatasetError


class DatasetBuilder:
    """
    Creates train, validation and test datasets.
    """

    def __init__(self):

        self.loader = AudioLoader()

        self.cleaner = AudioCleaner()

        random.seed(RANDOM_SEED)

    # ---------------------------------------------------

    def scan_dataset(self) -> Dict[str, List[Path]]:
        """
        Scan dataset folders.

        Returns
        -------
        Dictionary

        {
            "submarine": [...],
            "ship": [...],
            ...
        }
        """

        dataset = {}

        if not RAW_DATA_DIR.exists():
            raise DatasetError(f"{RAW_DATA_DIR} does not exist.")

        for folder in sorted(RAW_DATA_DIR.iterdir()):

            if not folder.is_dir():
                continue

            wav_files = sorted(folder.glob("*.wav"))

            if len(wav_files) == 0:
                logger.warning(f"No WAV files found in {folder.name}")
                continue

            dataset[folder.name] = wav_files

        return dataset

    # ---------------------------------------------------

    def encode_labels(
        self,
        dataset: Dict[str, List[Path]]
    ) -> Dict[str, int]:

        classes = sorted(dataset.keys())

        return {
            label: index
            for index, label in enumerate(classes)
        }
    
    # ---------------------------------------------------

    def save_metadata(
        self,
        label_encoder: Dict[str, int],
        X_train,
        X_val,
        X_test,
    ) -> None:
        """
        Save dataset metadata.
        """

        METADATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        class_mapping = {
            str(index): label
            for label, index in label_encoder.items()
        }

        dataset_info = {

            "num_classes": len(label_encoder),

            "train_samples": len(X_train),

            "validation_samples": len(X_val),

            "test_samples": len(X_test),

        }

        with open(
            METADATA_DIR / "class_mapping.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                class_mapping,
                file,
                indent=4,
            )

        with open(
            METADATA_DIR / "dataset_info.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                dataset_info,
                file,
                indent=4,
            )

        logger.info(
            "Metadata generated successfully."
        )

    # ---------------------------------------------------

    def build(self):

        logger.info("Scanning dataset...")

        dataset = self.scan_dataset()

        label_encoder = self.encode_labels(dataset)

        X = []

        y = []

        for label, wav_files in dataset.items():

            logger.info(
                f"Processing {label} ({len(wav_files)} files)"
            )

            for wav_file in wav_files:

                try:

                    waveform, _ = self.loader.load_audio(
                        wav_file
                    )

                    waveform = self.cleaner.clean(
                        waveform
                    )

                    X.append(waveform)

                    y.append(label_encoder[label])

                except Exception as error:

                    logger.error(
                        f"Failed loading {wav_file.name}"
                    )

                    logger.error(error)

        if len(X) == 0:
            raise DatasetError("Dataset is empty.")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            train_size=TRAIN_SPLIT,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=y,
        )

        validation_ratio = VALIDATION_SPLIT / (
            VALIDATION_SPLIT + TEST_SPLIT
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            train_size=validation_ratio,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=y_temp,
        )

        self.save_metadata(
            label_encoder,
            X_train,
            X_val,
            X_test,
        )

        logger.info("Dataset successfully built.")

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            label_encoder,
        )


if __name__ == "__main__":

    builder = DatasetBuilder()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        labels,
    ) = builder.build()

    print()

    print("========== DATASET SUMMARY ==========")

    print(f"Training Samples   : {len(X_train)}")

    print(f"Validation Samples : {len(X_val)}")

    print(f"Testing Samples    : {len(X_test)}")

    print()

    print("Label Mapping")

    print(labels)