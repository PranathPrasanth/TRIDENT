from pathlib import Path

# ==========================================================
# AUDIO CONFIGURATION
# ==========================================================

SAMPLE_RATE = 16000
MONO = True

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20

# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5
RANDOM_SEED = 42

CHECKPOINT_MONITOR = "val_accuracy"
EARLY_STOP_MONITOR = "val_loss"

# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

INPUT_HEIGHT = 128
INPUT_WIDTH = 128
INPUT_CHANNELS = 1

INPUT_SHAPE = (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    INPUT_CHANNELS,
)

# ==========================================================
# DATASET SPLITS
# ==========================================================

TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==========================================================
# PATHS
# ==========================================================

MODEL_DIR = "models"

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
FEATURE_DIR = Path("data/features")
SPECTROGRAM_DIR = Path("data/spectrograms")
METADATA_DIR = Path("data/metadata")

# ==========================================================
# LOGGING
# ==========================================================

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "trident.log"