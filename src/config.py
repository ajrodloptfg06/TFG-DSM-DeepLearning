"""Configuración de rutas para notebook, ejecución local y aplicación web."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configured_dir(environment_name: str, default_name: str) -> Path:
    configured = os.environ.get(environment_name)
    if configured:
        return Path(configured).expanduser().resolve()
    return (PROJECT_ROOT / default_name).resolve()


DATA_DIR = _configured_dir("TFG_DATA_DIR", "data")
RESULTS_DIR = _configured_dir("TFG_RESULTS_DIR", "results")
CHECKPOINTS_DIR = _configured_dir("TFG_CHECKPOINTS_DIR", "checkpoints")
WEB_RUNS_DIR = _configured_dir("TFG_WEB_RUNS_DIR", "web_runs")

DATA_PATHS = {
    "x_train": DATA_DIR / "xtrain_nyc-002.npy",
    "y_train": DATA_DIR / "ytrain_nyc.npy",
    "x_test": DATA_DIR / "xtest_nyc.npy",
    "y_test": DATA_DIR / "ytest_nyc.npy",
}

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_INPUT_SHAPE = (4, 128, 128)
DEFAULT_OUTPUT_SHAPE = (1, 128, 128)
