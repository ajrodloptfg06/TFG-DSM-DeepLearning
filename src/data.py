"""Carga reproducible de arrays DSM en formato NumPy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .config import DATA_PATHS


def _resolve_paths(paths: Mapping[str, str | Path] | None = None) -> dict[str, Path]:
    source = DATA_PATHS if paths is None else paths
    return {name: Path(path).expanduser().resolve() for name, path in source.items()}


def _require_files(paths: Mapping[str, Path], required: tuple[str, ...]) -> None:
    missing = [f"{name}: {paths[name]}" for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            "No se encontraron los archivos .npy requeridos. Configura TFG_DATA_DIR "
            "o coloca los datos en data/. Faltan: " + "; ".join(missing)
        )


def _to_nchw_float32(array: np.ndarray, *, expected_channels: int, name: str):
    import torch

    if array.ndim != 4:
        raise ValueError(f"{name} debe ser NHWC de cuatro dimensiones; shape={array.shape}")
    if array.shape[-1] != expected_channels:
        raise ValueError(
            f"{name} debe tener {expected_channels} canales en el último eje; shape={array.shape}"
        )
    if tuple(array.shape[1:3]) != (128, 128):
        raise ValueError(f"{name} debe contener patches 128x128; shape={array.shape}")
    return torch.from_numpy(array.astype(np.float32)).permute(0, 3, 1, 2)


def load_complete_dataset(
    paths: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    resolved = _resolve_paths(paths)
    required = ("x_train", "y_train", "x_test", "y_test")
    _require_files(resolved, required)

    arrays = {name: np.load(resolved[name]) for name in required}
    return {
        "x_train": _to_nchw_float32(arrays["x_train"], expected_channels=4, name="x_train"),
        "y_train": _to_nchw_float32(arrays["y_train"], expected_channels=1, name="y_train"),
        "x_test": _to_nchw_float32(arrays["x_test"], expected_channels=4, name="x_test"),
        "y_test": _to_nchw_float32(arrays["y_test"], expected_channels=1, name="y_test"),
        "paths": resolved,
    }


def load_demo_dataset(
    max_samples: int | None = 128,
    paths: Mapping[str, str | Path] | None = None,
):
    resolved = _resolve_paths(paths)
    required = ("x_train", "y_train")
    _require_files(resolved, required)

    if max_samples is not None and int(max_samples) < 2:
        raise ValueError("max_samples debe ser None o al menos 2.")

    x_train = np.load(resolved["x_train"], mmap_mode="r")
    y_train = np.load(resolved["y_train"], mmap_mode="r")
    if len(x_train) != len(y_train):
        raise ValueError(
            f"x_train e y_train tienen distinto número de muestras: {len(x_train)} != {len(y_train)}"
        )

    sample_count = len(x_train) if max_samples is None else min(int(max_samples), len(x_train))
    if sample_count < 2:
        raise ValueError("El dataset demo necesita al menos dos muestras.")

    x_demo = np.asarray(x_train[:sample_count])
    y_demo = np.asarray(y_train[:sample_count])
    return (
        _to_nchw_float32(x_demo, expected_channels=4, name="x_train demo"),
        _to_nchw_float32(y_demo, expected_channels=1, name="y_train demo"),
    )
