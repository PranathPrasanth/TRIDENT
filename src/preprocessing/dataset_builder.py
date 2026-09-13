"""
TRIDENT Dataset Builder

Builds machine learning datasets from underwater
acoustic recordings.

Pipeline:

WAV recording
    ↓
Train / Validation / Test recording split
    ↓
3-second acoustic segmentation
    ↓
Audio cleaning
    ↓
Mel spectrogram
    ↓
Normalization
    ↓
Fixed-size tensor
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

    Important:
    Dataset splitting happens at the RECORDING level
    before 3-second segmentation.

    This prevents segments from the same original recording
    from appearing in both training and testing.
    """

    # ---------------------------------------------------------
    # Acoustic segmentation configuration
    # ---------------------------------------------------------

    SEGMENT_DURATION = 3.0

    TARGET_HEIGHT = 128
    TARGET_WIDTH = 128

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def __init__(self) -> None:

        self.loader = AudioLoader()

        self.cleaner = AudioCleaner()

        self.extractor = MelSpectrogramExtractor()

        self.normalizer = FeatureNormalizer()

        random.seed(RANDOM_SEED)

    # =========================================================
    # Scan Dataset
    # =========================================================

    def scan_dataset(
        self,
    ) -> Dict[str, list[Path]]:
        """
        Scan dataset folders.

        Returns
        -------
        dict
            {
                "cargo_ship": [...],
                "passenger_ship": [...],
                "tanker": [...],
                "tug": [...]
            }
        """

        logger.info(
            "Scanning dataset..."
        )

        dataset: Dict[str, list[Path]] = {}

        if not RAW_DATA_DIR.exists():

            raise DatasetError(
                f"{RAW_DATA_DIR} does not exist."
            )

        for folder in sorted(
            RAW_DATA_DIR.iterdir()
        ):

            if not folder.is_dir():
                continue

            wav_files = sorted(
                folder.glob("*.wav")
            )

            if len(wav_files) == 0:

                logger.warning(
                    f"No WAV files found in "
                    f"{folder.name}"
                )

                continue

            dataset[
                folder.name
            ] = wav_files

            logger.info(
                f"{folder.name}: "
                f"{len(wav_files)} recordings"
            )

        if not dataset:

            raise DatasetError(
                "No valid WAV recordings found."
            )

        return dataset

    # =========================================================
    # Encode Labels
    # =========================================================

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

        label_encoder = {
            label: index
            for index, label
            in enumerate(classes)
        }

        logger.info(
            f"Detected classes: {classes}"
        )

        logger.info(
            f"Label mapping: {label_encoder}"
        )

        return label_encoder

    # =========================================================
    # Split Recordings
    # =========================================================

    def split_recordings(
        self,
        dataset: dict[str, list[Path]],
    ) -> tuple[
        dict[str, list[Path]],
        dict[str, list[Path]],
        dict[str, list[Path]],
    ]:
        """
        Split ORIGINAL RECORDINGS into train,
        validation and test sets.

        Splitting occurs before segmentation.

        Every class receives recordings in all three
        partitions whenever enough recordings exist.
        """

        train_files: dict[
            str, list[Path]
        ] = {}

        validation_files: dict[
            str, list[Path]
        ] = {}

        test_files: dict[
            str, list[Path]
        ] = {}

        logger.info(
            "Splitting recordings by class..."
        )

        for class_name in sorted(
            dataset.keys()
        ):

            files = list(
                dataset[class_name]
            )

            rng = np.random.default_rng(
                RANDOM_SEED
            )

            rng.shuffle(files)

            number_of_files = len(files)

            if number_of_files < 3:

                raise DatasetError(
                    f"Class '{class_name}' has only "
                    f"{number_of_files} recordings. "
                    f"At least 3 recordings are required "
                    f"for train/validation/test."
                )

            # -------------------------------------------------
            # Exactly three recordings
            # -------------------------------------------------

            if number_of_files == 3:

                train_count = 1
                validation_count = 1
                test_count = 1

            else:

                train_count = max(
                    1,
                    int(
                        number_of_files
                        * TRAIN_SPLIT
                    ),
                )

                validation_count = max(
                    1,
                    int(
                        number_of_files
                        * VALIDATION_SPLIT
                    ),
                )

                test_count = max(
                    1,
                    int(
                        number_of_files
                        * TEST_SPLIT
                    ),
                )

                total = (
                    train_count
                    + validation_count
                    + test_count
                )

                # ---------------------------------------------
                # Give remaining recordings to training
                # ---------------------------------------------

                while (
                    total < number_of_files
                ):

                    train_count += 1
                    total += 1

                # ---------------------------------------------
                # Correct rounding overflow
                # ---------------------------------------------

                while (
                    total > number_of_files
                ):

                    if (
                        train_count
                        > 1
                        and train_count
                        >= validation_count
                        and train_count
                        >= test_count
                    ):

                        train_count -= 1

                    elif (
                        validation_count > 1
                        and validation_count
                        >= test_count
                    ):

                        validation_count -= 1

                    elif test_count > 1:

                        test_count -= 1

                    else:

                        raise DatasetError(
                            "Unable to create a valid "
                            "recording split."
                        )

                    total = (
                        train_count
                        + validation_count
                        + test_count
                    )

            # -------------------------------------------------
            # Assign files
            # -------------------------------------------------

            train_end = train_count

            validation_end = (
                train_end
                + validation_count
            )

            train_files[class_name] = files[
                :train_end
            ]

            validation_files[class_name] = files[
                train_end:validation_end
            ]

            test_files[class_name] = files[
                validation_end:
            ]

            logger.info(
                f"{class_name}: "
                f"{number_of_files} recordings -> "
                f"Train={len(train_files[class_name])}, "
                f"Validation={len(validation_files[class_name])}, "
                f"Test={len(test_files[class_name])}"
            )

        return (
            train_files,
            validation_files,
            test_files,
        )

    # =========================================================
    # Segment Audio
    # =========================================================

    def segment_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> list[np.ndarray]:
        """
        Divide an audio recording into non-overlapping
        3-second segments.

        The final incomplete segment is discarded.

        Returns
        -------
        list[np.ndarray]
            List of audio segments.
        """

        samples_per_segment = int(
            self.SEGMENT_DURATION
            * sample_rate
        )

        if samples_per_segment <= 0:

            raise DatasetError(
                "Invalid segment duration."
            )

        total_samples = len(
            waveform
        )

        segments: list[np.ndarray] = []

        start = 0

        while (
            start + samples_per_segment
            <= total_samples
        ):

            end = (
                start
                + samples_per_segment
            )

            segment = waveform[
                start:end
            ]

            segments.append(
                segment
            )

            start = end

        return segments

    # =========================================================
    # Resize Mel Spectrogram
    # =========================================================

    def resize_mel(
        self,
        mel: np.ndarray,
        target_height: int = TARGET_HEIGHT,
        target_width: int = TARGET_WIDTH,
    ) -> np.ndarray:
        """
        Convert a Mel spectrogram into a fixed size.

        Target shape:
            (128, 128)
        """

        height, width = mel.shape

        # -----------------------------------------------------
        # Fix Mel-frequency dimension
        # -----------------------------------------------------

        if height > target_height:

            mel = mel[
                :target_height,
                :
            ]

        elif height < target_height:

            padding = (
                target_height
                - height
            )

            mel = np.pad(
                mel,
                (
                    (0, padding),
                    (0, 0),
                ),
                mode="constant",
            )

        # -----------------------------------------------------
        # Fix time dimension
        # -----------------------------------------------------

        if width > target_width:

            mel = mel[
                :,
                :target_width,
            ]

        elif width < target_width:

            padding = (
                target_width
                - width
            )

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

    # =========================================================
    # Process One Recording
    # =========================================================

    def process_recording(
        self,
        wav_file: Path,
        label: int,
    ) -> tuple[
        list[np.ndarray],
        list[int],
    ]:
        """
        Load one recording, clean it, segment it and
        convert each segment into a Mel-spectrogram tensor.
        """

        try:

            # -------------------------------------------------
            # Load
            # -------------------------------------------------

            waveform, sample_rate = (
                self.loader.load_audio(
                    wav_file
                )
            )

            # -------------------------------------------------
            # Clean
            # -------------------------------------------------

            waveform = (
                self.cleaner.clean(
                    waveform
                )
            )

            # -------------------------------------------------
            # Segment
            # -------------------------------------------------

            segments = self.segment_audio(
                waveform,
                sample_rate,
            )

            if not segments:

                logger.warning(
                    f"{wav_file.name}: "
                    f"recording shorter than "
                    f"{self.SEGMENT_DURATION}s. "
                    f"Skipping."
                )

                return [], []

            features: list[np.ndarray] = []
            labels: list[int] = []

            # -------------------------------------------------
            # Process every segment
            # -------------------------------------------------

            for segment_index, segment in enumerate(
                segments
            ):

                try:

                    # -----------------------------------------
                    # Mel spectrogram
                    # -----------------------------------------

                    mel = (
                        self.extractor.extract(
                            segment
                        )
                    )

                    # -----------------------------------------
                    # Normalize
                    # -----------------------------------------

                    mel = (
                        self.normalizer.normalize(
                            mel
                        )
                    )

                    # -----------------------------------------
                    # Fixed size
                    # -----------------------------------------

                    mel = self.resize_mel(
                        mel
                    )

                    # -----------------------------------------
                    # Channel dimension
                    # -----------------------------------------

                    mel = np.expand_dims(
                        mel,
                        axis=-1,
                    )

                    features.append(
                        mel
                    )

                    labels.append(
                        label
                    )

                except Exception as error:

                    logger.warning(
                        f"Failed processing segment "
                        f"{segment_index} from "
                        f"{wav_file.name}: "
                        f"{error}"
                    )

            logger.info(
                f"{wav_file.name}: "
                f"{len(features)} segments generated."
            )

            return (
                features,
                labels,
            )

        except Exception as error:

            logger.error(
                f"Failed loading "
                f"{wav_file.name}: "
                f"{error}"
            )

            return [], []

    # =========================================================
    # Build Partition
    # =========================================================

    def build_partition(
        self,
        partition: dict[str, list[Path]],
        label_encoder: dict[str, int],
        partition_name: str,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Convert a recording partition into Mel-spectrogram
        tensors.
        """

        logger.info(
            f"Building {partition_name} partition..."
        )

        X: list[np.ndarray] = []
        y: list[int] = []

        recording_count = 0

        for class_name in sorted(
            partition.keys()
        ):

            wav_files = partition[
                class_name
            ]

            label = label_encoder[
                class_name
            ]

            logger.info(
                f"{partition_name} | "
                f"{class_name}: "
                f"{len(wav_files)} recordings"
            )

            for wav_file in wav_files:

                recording_count += 1

                features, labels = (
                    self.process_recording(
                        wav_file,
                        label,
                    )
                )

                X.extend(
                    features
                )

                y.extend(
                    labels
                )

        if not X:

            raise DatasetError(
                f"{partition_name} partition "
                f"is empty."
            )

        X_array = np.asarray(
            X,
            dtype=np.float32,
        )

        y_array = np.asarray(
            y,
            dtype=np.int32,
        )

        logger.info(
            f"{partition_name} partition completed: "
            f"{recording_count} recordings -> "
            f"{len(X_array)} segments"
        )

        logger.info(
            f"{partition_name} shape: "
            f"{X_array.shape}"
        )

        return (
            X_array,
            y_array,
        )

    # =========================================================
    # Save Metadata
    # =========================================================

    def save_metadata(
        self,
        label_encoder: Dict[str, int],
        train_files: dict[str, list[Path]],
        validation_files: dict[str, list[Path]],
        test_files: dict[str, list[Path]],
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

            "classes": sorted(
                label_encoder.keys()
            ),

            "segment_duration_seconds":
                self.SEGMENT_DURATION,

            "input_shape": [
                self.TARGET_HEIGHT,
                self.TARGET_WIDTH,
                1,
            ],

            "train_recordings": sum(
                len(files)
                for files
                in train_files.values()
            ),

            "validation_recordings": sum(
                len(files)
                for files
                in validation_files.values()
            ),

            "test_recordings": sum(
                len(files)
                for files
                in test_files.values()
            ),

            "train_segments": len(
                X_train
            ),

            "validation_segments": len(
                X_val
            ),

            "test_segments": len(
                X_test
            ),

            "segment_split_strategy":
                "recording_level_before_segmentation",
        }

        # -----------------------------------------------------
        # Save class mapping
        # -----------------------------------------------------

        with open(
            METADATA_DIR
            / "class_mapping.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                class_mapping,
                file,
                indent=4,
            )

        # -----------------------------------------------------
        # Save dataset information
        # -----------------------------------------------------

        with open(
            METADATA_DIR
            / "dataset_info.json",
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

    # =========================================================
    # Build Complete Dataset
    # =========================================================

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
        """
        Build the complete train, validation and
        test dataset.

        Returns
        -------
        tuple
            X_train
            X_val
            X_test
            y_train
            y_val
            y_test
            label_encoder
        """

        logger.info(
            "Starting TRIDENT dataset build..."
        )

        # -----------------------------------------------------
        # Scan
        # -----------------------------------------------------

        dataset = self.scan_dataset()

        # -----------------------------------------------------
        # Labels
        # -----------------------------------------------------

        label_encoder = (
            self.encode_labels(
                dataset
            )
        )

        # -----------------------------------------------------
        # Split ORIGINAL RECORDINGS
        # -----------------------------------------------------

        (
            train_files,
            validation_files,
            test_files,
        ) = self.split_recordings(
            dataset
        )

        # -----------------------------------------------------
        # Build train partition
        # -----------------------------------------------------

        (
            X_train,
            y_train,
        ) = self.build_partition(
            train_files,
            label_encoder,
            "TRAIN",
        )

        # -----------------------------------------------------
        # Build validation partition
        # -----------------------------------------------------

        (
            X_val,
            y_val,
        ) = self.build_partition(
            validation_files,
            label_encoder,
            "VALIDATION",
        )

        # -----------------------------------------------------
        # Build test partition
        # -----------------------------------------------------

        (
            X_test,
            y_test,
        ) = self.build_partition(
            test_files,
            label_encoder,
            "TEST",
        )

        # -----------------------------------------------------
        # Save metadata
        # -----------------------------------------------------

        self.save_metadata(
            label_encoder,
            train_files,
            validation_files,
            test_files,
            X_train,
            X_val,
            X_test,
        )

        # -----------------------------------------------------
        # Final summary
        # -----------------------------------------------------

        logger.info(
            "======================================"
        )

        logger.info(
            "TRIDENT DATASET BUILD COMPLETE"
        )

        logger.info(
            f"Train segments      : {len(X_train)}"
        )

        logger.info(
            f"Validation segments : {len(X_val)}"
        )

        logger.info(
            f"Test segments       : {len(X_test)}"
        )

        logger.info(
            f"Input shape         : {X_train.shape[1:]}"
        )

        logger.info(
            "======================================"
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            label_encoder,
        )


# =============================================================
# Standalone Test
# =============================================================

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
        "========== TRIDENT DATASET SUMMARY =========="
    )

    print(
        f"Training Segments   : {len(X_train)}"
    )

    print(
        f"Validation Segments : {len(X_val)}"
    )

    print(
        f"Testing Segments    : {len(X_test)}"
    )

    print()

    print(
        "Label Mapping"
    )

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