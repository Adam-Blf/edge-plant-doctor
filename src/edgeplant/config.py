"""Shared constants: paths, input geometry, normalisation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

INPUT_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 38
SEED = 42
