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

from src.preprocessing.dataset_builder import DatasetBuilder

from src.training.dataset import DatasetManager
from src.training.model import TRIDENTModel
from src.training.trainer import ModelTrainer
from src.training.evaluate import ModelEvaluator

from src.inference.predictor import Predictor

from src.threat_engine.analyzer import ThreatAnalyzer

from src.explainability.gradcam import GradCAM

from src.utils.logger import logger


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

    logger.info("=" * 60)

    logger.info(
        "TRIDENT Training Pipeline Started"
    )

    logger.info("=" * 60)

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

    class_weights = trainer.calculate_class_weights(
        y_train
    )

    logger.info(
        "Class weights calculated."
    )

    trainer.train(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        class_weights=class_weights,
    )

    logger.info(
        "Training finished."
    )

    # --------------------------------------------------
    # Training Set Sanity Check
    # --------------------------------------------------
    #
    # This diagnostic deliberately bypasses tf.data
    # evaluation so that we can inspect the model's
    # predictions directly against X_train and y_train.
    #
    # This helps determine whether the model is actually
    # learning the training data or collapsing to one class.
    # --------------------------------------------------

    logger.info("=" * 60)

    logger.info(
        "TRAINING SET SANITY CHECK"
    )

    logger.info("=" * 60)

    # Direct NumPy prediction.
    #
    # We intentionally use X_train directly instead of
    # train_dataset so that dataset shuffling does not
    # affect this diagnostic.

    train_probabilities = model.predict(
        X_train,
        batch_size=32,
        verbose=0,
    )

    train_predictions = np.argmax(
        train_probabilities,
        axis=1,
    )

    print()
    print("=" * 60)
    print("TRAINING SET SANITY CHECK")
    print("=" * 60)

    # --------------------------------------------------
    # Actual Training Distribution
    # --------------------------------------------------

    print()
    print("Actual training class distribution:")

    actual_classes, actual_counts = np.unique(
        y_train,
        return_counts=True,
    )

    for class_id, count in zip(
        actual_classes,
        actual_counts,
    ):

        class_name = next(
            (
                name
                for name, index
                in label_encoder.items()
                if index == class_id
            ),
            f"class_{class_id}",
        )

        print(
            f"  {class_id} ({class_name}): "
            f"{count}"
        )

    # --------------------------------------------------
    # Predicted Training Distribution
    # --------------------------------------------------

    print()
    print("Predicted training class distribution:")

    predicted_classes, predicted_counts = np.unique(
        train_predictions,
        return_counts=True,
    )

    for class_id, count in zip(
        predicted_classes,
        predicted_counts,
    ):

        class_name = next(
            (
                name
                for name, index
                in label_encoder.items()
                if index == class_id
            ),
            f"class_{class_id}",
        )

        print(
            f"  {class_id} ({class_name}): "
            f"{count}"
        )

    # --------------------------------------------------
    # Direct Training Accuracy
    # --------------------------------------------------

    train_accuracy = np.mean(
        train_predictions == y_train
    )

    print()
    print(
        f"Train accuracy : "
        f"{train_accuracy:.4%}"
    )

    print("=" * 60)

    logger.info(
        "Training set sanity check completed."
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

    return (
        model,
        X_test,
        y_test,
        label_encoder,
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

    logger.info("=" * 60)

    logger.info(
        "TRIDENT Inference Pipeline Started"
    )

    logger.info("=" * 60)

    predictor = Predictor()

    prediction, confidence = predictor.predict(
        audio_path
    )

    print()

    print("========== PREDICTION ==========")

    print(
        f"Target      : {prediction}"
    )

    print(
        f"Confidence  : {confidence:.2%}"
    )

    # --------------------------------------------------
    # Threat Analysis
    # --------------------------------------------------

    analyzer = ThreatAnalyzer()

    result = analyzer.analyze(
        prediction,
        confidence,
    )

    print()

    print(
        "========== THREAT ANALYSIS =========="
    )

    print(
        f"Category            : "
        f"{result['category']}"
    )

    print(
        f"Threat Level        : "
        f"{result['threat_level']}"
    )

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
                    f"Audio file not found: "
                    f"{args.predict}"
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