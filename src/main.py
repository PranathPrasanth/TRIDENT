"""
TRIDENT Main Pipeline

Orchestrates the complete underwater acoustic
classification pipeline.

Pipeline
--------
Dataset Builder
    ↓
TensorFlow Dataset
    ↓
CNN Training
    ↓
Evaluation
    ↓
Inference (Optional)
    ↓
Threat Analysis
    ↓
Grad-CAM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from preprocessing.dataset_builder import DatasetBuilder

from training.dataset import DatasetManager
from training.model import TRIDENTModel
from training.trainer import ModelTrainer
from training.evaluate import ModelEvaluator

from inference.predictor import Predictor

from threat_engine.analyzer import ThreatAnalyzer

from explainability.gradcam import GradCAM

from utils.logger import logger


# ==========================================================
# Training Pipeline
# ==========================================================

def train_pipeline():
    """
    Build dataset, train the CNN and evaluate it.

    Returns
    -------
    tuple
        (
            trained_model,
            X_test,
            y_test,
            label_encoder,
        )
    """

    logger.info(
        "=" * 60
    )

    logger.info(
        "TRIDENT Training Pipeline Started"
    )

    logger.info(
        "=" * 60
    )

    # --------------------------------------------------
    # Dataset Builder
    # --------------------------------------------------

    builder = DatasetBuilder()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_encoder,
    ) = builder.build()

    logger.info(
        "Dataset built successfully."
    )

    # --------------------------------------------------
    # TensorFlow Dataset
    # --------------------------------------------------

    dataset_manager = DatasetManager()

    train_dataset = dataset_manager.create_dataset(
        X_train,
        y_train,
        training=True,
    )

    validation_dataset = dataset_manager.create_dataset(
        X_val,
        y_val,
        training=False,
    )

    logger.info(
        "TensorFlow datasets created."
    )

    # --------------------------------------------------
    # Build Model
    # --------------------------------------------------

    cnn = TRIDENTModel(
        num_classes=len(label_encoder)
    )

    model = cnn.build()

    logger.info(
        "CNN initialized."
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    trainer = ModelTrainer()

    trainer.train(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    logger.info(
        "Training finished."
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    evaluator = ModelEvaluator(model)

    evaluator.evaluate(
        X_test,
        y_test,
    )

    logger.info(
        "Evaluation finished."
    )



# ==========================================================
# Inference Pipeline
# ==========================================================

def inference_pipeline(
    audio_path: str | Path,
):
    """
    Run inference on a new audio file.
    """

    logger.info(
        "=" * 60
    )

    logger.info(
        "TRIDENT Inference Pipeline Started"
    )

    logger.info(
        "=" * 60
    )

    predictor = Predictor()

    prediction, confidence = predictor.predict(
        audio_path
    )

    print()

    print("========== PREDICTION ==========")

    print(f"Target      : {prediction}")

    print(f"Confidence  : {confidence:.2%}")

    # --------------------------------------------------
    # Threat Analysis
    # --------------------------------------------------

    analyzer = ThreatAnalyzer()

    result = analyzer.analyze(
        prediction,
        confidence,
    )

    print()

    print("========== THREAT ANALYSIS ==========")

    print(f"Category            : {result['category']}")

    print(f"Threat Level        : {result['threat_level']}")

    print(
        f"Recommended Action  : "
        f"{result['recommended_action']}"
    )

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------

    waveform, _ = predictor.loader.load_audio(
        audio_path
    )

    waveform = predictor.cleaner.clean(
        waveform
    )

    mel = predictor.extractor.extract(
        waveform
    )

    mel = predictor.normalizer.normalize(
        mel
    )

    mel = np.expand_dims(
        mel,
        axis=-1,
    )

    mel = np.expand_dims(
        mel,
        axis=0,
    )

    # Ensure model graph is built
    predictor.model.predict(
        mel,
        verbose=0,
    )

    gradcam = GradCAM(
        predictor.model
    )

    heatmap = gradcam.generate(
        mel
    )

    logger.info(
        f"Grad-CAM generated "
        f"(shape={heatmap.shape})."
    )

# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    TRIDENT application entry point.
    """

    parser = argparse.ArgumentParser(

        description=(
            "TRIDENT - Underwater Acoustic "
            "Intelligence Platform"
        )

    )

    parser.add_argument(

        "--predict",

        type=Path,

        help=(
            "Path to an audio file for inference."
        ),

    )

    args = parser.parse_args()

    try:

        # ----------------------------------------------
        # Inference Mode
        # ----------------------------------------------

        if args.predict is not None:

            if not args.predict.exists():

                raise FileNotFoundError(

                    f"Audio file not found: {args.predict}"

                )

            inference_pipeline(
                args.predict
            )

        # ----------------------------------------------
        # Training Mode
        # ----------------------------------------------

        else:

            train_pipeline()

        logger.info(
            "TRIDENT execution completed successfully."
        )

    except KeyboardInterrupt:

        logger.warning(
            "Execution interrupted by user."
        )

        sys.exit(1)

    except Exception as error:

        logger.exception(
            "TRIDENT execution failed."
        )

        print()

        print("Execution failed.")

        print(error)

        sys.exit(1)
    
# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    TRIDENT application entry point.
    """

    parser = argparse.ArgumentParser(

        description=(
            "TRIDENT - Underwater Acoustic "
            "Intelligence Platform"
        )

    )

    parser.add_argument(

        "--predict",

        type=Path,

        help=(
            "Path to an audio file for inference."
        ),

    )

    args = parser.parse_args()

    try:

        # ----------------------------------------------
        # Inference Mode
        # ----------------------------------------------

        if args.predict is not None:

            if not args.predict.exists():

                raise FileNotFoundError(

                    f"Audio file not found: {args.predict}"

                )

            inference_pipeline(
                args.predict
            )

        # ----------------------------------------------
        # Training Mode
        # ----------------------------------------------

        else:

            train_pipeline()

        logger.info(
            "TRIDENT execution completed successfully."
        )

    except KeyboardInterrupt:

        logger.warning(
            "Execution interrupted by user."
        )

        sys.exit(1)

    except Exception as error:

        logger.exception(
            "TRIDENT execution failed."
        )

        print()

        print("Execution failed.")

        print(error)

        sys.exit(1)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()