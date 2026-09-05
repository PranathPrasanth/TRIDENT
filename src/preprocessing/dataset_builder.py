"""
TRIDENT Dataset Builder

Builds machine learning datasets from underwater
acoustic recordings.
"""

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np

from src.utils.config import (
    METADATA_DIR,
    RAW_DATA_DIR,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)

from src.feature_extraction.mel_spectrogram import (
    MelSpectrogramExtractor,
)

from src.feature_extraction.normalizer import (
    FeatureNormalizer,
)

from src.preprocessing.audio_loader import (
    AudioLoader,
)

from src.preprocessing.audio_cleaner import (
    AudioCleaner,
)

from src.utils.logger import logger

from src.utils.exceptions import (
    DatasetError,
)


class DatasetBuilder:
    """
    Creates train, validation and test datasets.
    """

    def __init__(self) -> None:

        self.loader = AudioLoader()

        self.cleaner = AudioCleaner()

        self.extractor = MelSpectrogramExtractor()

        self.normalizer = FeatureNormalizer()

        random.seed(RANDOM_SEED)

    # ---------------------------------------------------
    # Scan Dataset
    # ---------------------------------------------------

    def scan_dataset(self) -> Dict[str, list[Path]]:
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
            raise DatasetError(
                f"{RAW_DATA_DIR} does not exist."
            )

        for folder in sorted(RAW_DATA_DIR.iterdir()):

            if not folder.is_dir():
                continue

            wav_files = sorted(
                folder.glob("*.wav")
            )

            if len(wav_files) == 0:

                logger.warning(
                    f"No WAV files found in {folder.name}"
                )

                continue

            dataset[folder.name] = wav_files

        return dataset

    # ---------------------------------------------------
    # Encode Labels
    # ---------------------------------------------------

    def encode_labels(
        self,
        dataset: dict[str, list[Path]],
    ) -> dict[str, int]:
        """
        Create a numerical label for every class.
        """

        classes = sorted(
            dataset.keys()
        )

        return {
            label: index
            for index, label in enumerate(classes)
        }

    # ---------------------------------------------------
    # Resize Mel Spectrogram
    # ---------------------------------------------------

    def resize_mel(
        self,
        mel: np.ndarray,
        target_height: int = 128,
        target_width: int = 128,
    ) -> np.ndarray:
        """
        Convert a Mel spectrogram into a fixed size.

        Every spectrogram must have the same dimensions
        before it can be combined into one NumPy array.

        Target shape:

            (128, 128)
        """

        height, width = mel.shape

        # ------------------------------------------------
        # Fix Mel-frequency dimension
        # ------------------------------------------------

        if height > target_height:

            mel = mel[
                :target_height,
                :
            ]

        elif height < target_height:

            padding = target_height - height

            mel = np.pad(
                mel,
                (
                    (0, padding),
                    (0, 0),
                ),
                mode="constant",
            )

        # ------------------------------------------------
        # Fix time dimension
        # ------------------------------------------------

        if width > target_width:

            mel = mel[
                :,
                :target_width
            ]

        elif width < target_width:

            padding = target_width - width

            mel = np.pad(
                mel,
                (
                    (0, 0),
                    (0, padding),
                ),
                mode="constant",
            )

        return mel.astype(
            np.float32
        )

    # ---------------------------------------------------
    # Split Dataset By Class
    # ---------------------------------------------------

    def split_by_class(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Split the dataset into train, validation and test
        sets while making sure every class appears in
        every split.

        This is especially important for small datasets.

        Example:

        If a class has only 3 samples:

            Train = 1
            Validation = 1
            Test = 1

        This avoids failures caused by stratified
        train_test_split when a class has very few samples.
        """

        train_indices = []
        validation_indices = []
        test_indices = []

        # ------------------------------------------------
        # Process each class separately
        # ------------------------------------------------

        classes = np.unique(y)

        for class_label in classes:

            class_indices = np.where(
                y == class_label
            )[0]

            # Shuffle class samples
            rng = np.random.default_rng(
                RANDOM_SEED + int(class_label)
            )

            rng.shuffle(
                class_indices
            )

            number_of_samples = len(
                class_indices
            )

            # ------------------------------------------------
            # Need at least 3 samples
            # ------------------------------------------------

            if number_of_samples < 3:

                raise DatasetError(
                    f"Class {class_label} has only "
                    f"{number_of_samples} samples. "
                    "At least 3 samples are required "
                    "to create train, validation and "
                    "test sets."
                )

            # ------------------------------------------------
            # Give every class one sample in every split
            # ------------------------------------------------

            if number_of_samples == 3:

                train_count = 1
                validation_count = 1
                test_count = 1

            else:

                # Start with the requested proportions
                train_count = max(
                    1,
                    int(
                        number_of_samples
                        * TRAIN_SPLIT
                    ),
                )

                validation_count = max(
                    1,
                    int(
                        number_of_samples
                        * VALIDATION_SPLIT
                    ),
                )

                test_count = max(
                    1,
                    int(
                        number_of_samples
                        * TEST_SPLIT
                    ),
                )

                # ------------------------------------------------
                # Correct rounding differences
                # ------------------------------------------------

                total = (
                    train_count
                    + validation_count
                    + test_count
                )

                while total < number_of_samples:

                    train_count += 1
                    total += 1

                while total > number_of_samples:

                    # Remove from the largest split
                    # but never reduce below 1.
                    if (
                        train_count
                        >= validation_count
                        and train_count
                        >= test_count
                        and train_count > 1
                    ):
                        train_count -= 1

                    elif (
                        validation_count
                        >= test_count
                        and validation_count > 1
                    ):
                        validation_count -= 1

                    elif test_count > 1:
                        test_count -= 1

                    total = (
                        train_count
                        + validation_count
                        + test_count
                    )

            # ------------------------------------------------
            # Create index ranges
            # ------------------------------------------------

            train_end = train_count

            validation_end = (
                train_end
                + validation_count
            )

            train_indices.extend(
                class_indices[
                    :train_end
                ]
            )

            validation_indices.extend(
                class_indices[
                    train_end:validation_end
                ]
            )

            test_indices.extend(
                class_indices[
                    validation_end:
                ]
            )

            logger.info(
                f"Class {class_label}: "
                f"{number_of_samples} samples -> "
                f"Train={train_count}, "
                f"Validation={validation_count}, "
                f"Test={test_count}"
            )

        # ------------------------------------------------
        # Convert indices to arrays
        # ------------------------------------------------

        train_indices = np.asarray(
            train_indices,
            dtype=np.int32,
        )

        validation_indices = np.asarray(
            validation_indices,
            dtype=np.int32,
        )

        test_indices = np.asarray(
            test_indices,
            dtype=np.int32,
        )

        # ------------------------------------------------
        # Shuffle final splits
        # ------------------------------------------------

        rng = np.random.default_rng(
            RANDOM_SEED
        )

        rng.shuffle(train_indices)
        rng.shuffle(validation_indices)
        rng.shuffle(test_indices)

        # ------------------------------------------------
        # Create datasets
        # ------------------------------------------------

        X_train = X[
            train_indices
        ]

        y_train = y[
            train_indices
        ]

        X_val = X[
            validation_indices
        ]

        y_val = y[
            validation_indices
        ]

        X_test = X[
            test_indices
        ]

        y_test = y[
            test_indices
        ]

        logger.info(
            "Dataset split completed successfully."
        )

        logger.info(
            f"Train samples      : {len(X_train)}"
        )

        logger.info(
            f"Validation samples : {len(X_val)}"
        )

        logger.info(
            f"Test samples       : {len(X_test)}"
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )

    # ---------------------------------------------------
    # Save Metadata
    # ---------------------------------------------------

    def save_metadata(
        self,
        label_encoder: Dict[str, int],
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
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
            for label, index
            in label_encoder.items()
        }

        dataset_info = {

            "num_classes": len(
                label_encoder
            ),

            "train_samples": len(
                X_train
            ),

            "validation_samples": len(
                X_val
            ),

            "test_samples": len(
                X_test
            ),

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
    # Build Dataset
    # ---------------------------------------------------

    def build(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, int],
    ]:

        logger.info(
            "Scanning dataset..."
        )

        dataset = self.scan_dataset()

        if not dataset:

            raise DatasetError(
                "No valid dataset classes found."
            )

        label_encoder = self.encode_labels(
            dataset
        )

        X: list[np.ndarray] = []

        y: list[int] = []

        # ------------------------------------------------
        # Process every class
        # ------------------------------------------------

        for label, wav_files in dataset.items():

            logger.info(
                f"Processing {label} "
                f"({len(wav_files)} files)"
            )

            for wav_file in wav_files:

                try:

                    # ------------------------------------
                    # Load audio
                    # ------------------------------------

                    waveform, _ = (
                        self.loader.load_audio(
                            wav_file
                        )
                    )

                    # ------------------------------------
                    # Clean audio
                    # ------------------------------------

                    waveform = (
                        self.cleaner.clean(
                            waveform
                        )
                    )

                    # ------------------------------------
                    # Generate Mel spectrogram
                    # ------------------------------------

                    mel = (
                        self.extractor.extract(
                            waveform
                        )
                    )

                    # ------------------------------------
                    # Normalize
                    # ------------------------------------

                    mel = (
                        self.normalizer.normalize(
                            mel
                        )
                    )

                    # ------------------------------------
                    # Make fixed size
                    # ------------------------------------

                    mel = self.resize_mel(
                        mel
                    )

                    # ------------------------------------
                    # Add channel dimension
                    # ------------------------------------

                    mel = np.expand_dims(
                        mel,
                        axis=-1,
                    )

                    # ------------------------------------
                    # Store sample
                    # ------------------------------------

                    X.append(
                        mel
                    )

                    y.append(
                        label_encoder[label]
                    )

                except Exception as error:

                    logger.error(
                        f"Failed loading "
                        f"{wav_file.name}"
                    )

                    logger.error(
                        error
                    )

        # ------------------------------------------------
        # Convert lists to NumPy arrays
        # ------------------------------------------------

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y = np.asarray(
            y,
            dtype=np.int32,
        )

        # ------------------------------------------------
        # Check dataset
        # ------------------------------------------------

        if len(X) == 0:

            raise DatasetError(
                "Dataset is empty."
            )

        logger.info(
            f"Dataset shape: {X.shape}"
        )

        logger.info(
            f"Labels shape: {y.shape}"
        )

        # ------------------------------------------------
        # Train / Validation / Test Split
        # ------------------------------------------------

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = self.split_by_class(
            X,
            y,
        )

        # ------------------------------------------------
        # Save Metadata
        # ------------------------------------------------

        self.save_metadata(
            label_encoder,
            X_train,
            X_val,
            X_test,
        )

        logger.info(
            "Dataset successfully built."
        )

        # ------------------------------------------------
        # Return Dataset
        # ------------------------------------------------

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            label_encoder,
        )


# =======================================================
# Standalone Test
# =======================================================

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

    print(
        "========== DATASET SUMMARY =========="
    )

    print(
        f"Training Samples   : {len(X_train)}"
    )

    print(
        f"Validation Samples : {len(X_val)}"
    )

    print(
        f"Testing Samples    : {len(X_test)}"
    )

    print()

    print("Label Mapping")

    print(labels)

    print()

    print(
        "Training Shape     : "
        f"{X_train.shape}"
    )

    print(
        "Validation Shape   : "
        f"{X_val.shape}"
    )

    print(
        "Testing Shape      : "
        f"{X_test.shape}"
    )