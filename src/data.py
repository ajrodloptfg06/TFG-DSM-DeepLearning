"""Carga reproducible de arrays DSM en formato NumPy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .config import DATA_PATHS


DEMO_DATA_PATH = Path(__file__).resolve().parents[1] / "demo_data.npz"


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


def _sample_to_nchw_float32(array: np.ndarray, *, expected_channels: int, name: str):
    """Convierte una única muestra NHWC o NCHW sin cargar el dataset completo."""
    import torch

    sample = np.asarray(array)
    if sample.ndim != 4 or sample.shape[0] != 1:
        raise ValueError(f"{name} debe contener una única muestra 4D; shape={sample.shape}")

    if sample.shape[-1] == expected_channels and tuple(sample.shape[1:3]) == (128, 128):
        return torch.from_numpy(sample.astype(np.float32)).permute(0, 3, 1, 2)
    if sample.shape[1] == expected_channels and tuple(sample.shape[2:4]) == (128, 128):
        return torch.from_numpy(sample.astype(np.float32))

    raise ValueError(
        f"{name} debe tener forma (1, 128, 128, {expected_channels}) o "
        f"(1, {expected_channels}, 128, 128); shape={sample.shape}"
    )


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

    if max_samples is not None and int(max_samples) < 2:
        raise ValueError("max_samples debe ser None o al menos 2.")

    if all(resolved[name].is_file() for name in required):
        x_train = np.load(resolved["x_train"], mmap_mode="r")
        y_train = np.load(resolved["y_train"], mmap_mode="r")
    elif DEMO_DATA_PATH.is_file():
        with np.load(DEMO_DATA_PATH, allow_pickle=False) as demo:
            missing_keys = {
                "x_train_demo", "y_train_demo"
            } - set(demo.files)
            if missing_keys:
                raise ValueError(
                    "El archivo demo no contiene los arrays de entrenamiento requeridos."
                )
            x_train = demo["x_train_demo"]
            y_train = demo["y_train_demo"]
    else:
        _require_files(resolved, required)

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


def get_demo_training_info(
    paths: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    """Informa de la fuente y capacidad disponibles para el entrenamiento web."""
    resolved = _resolve_paths(paths)
    required = ("x_train", "y_train")
    if all(resolved[name].is_file() for name in required):
        x_train = np.load(resolved["x_train"], mmap_mode="r")
        y_train = np.load(resolved["y_train"], mmap_mode="r")
        source = "complete"
    elif DEMO_DATA_PATH.is_file():
        with np.load(DEMO_DATA_PATH, allow_pickle=False) as demo:
            missing_keys = {
                "x_train_demo", "y_train_demo"
            } - set(demo.files)
            if missing_keys:
                raise ValueError(
                    "El archivo demo no contiene los arrays de entrenamiento requeridos."
                )
            x_train = demo["x_train_demo"]
            y_train = demo["y_train_demo"]
        source = "demo"
    else:
        _require_files(resolved, required)

    if len(x_train) != len(y_train):
        raise ValueError("Los arrays de entrenamiento disponibles no están emparejados.")
    if len(x_train) < 2:
        raise ValueError("El entrenamiento demo necesita al menos dos muestras.")
    return {"sample_count": len(x_train), "source": source}


def get_test_dataset_size(
    paths: Mapping[str, str | Path] | None = None,
) -> int:
    """Devuelve el número de muestras de test usando cabeceras/memmap de NumPy."""
    resolved = _resolve_paths(paths)
    required = ("x_test", "y_test")
    _require_files(resolved, required)

    x_test = np.load(resolved["x_test"], mmap_mode="r")
    y_test = np.load(resolved["y_test"], mmap_mode="r")
    if len(x_test) != len(y_test):
        raise ValueError(
            f"x_test e y_test tienen distinto número de muestras: {len(x_test)} != {len(y_test)}"
        )
    if len(x_test) < 1:
        raise ValueError("El conjunto de test está vacío.")
    return len(x_test)


def load_test_sample(
    index: int,
    paths: Mapping[str, str | Path] | None = None,
):
    """Carga solo una muestra emparejada de test y la devuelve en formato NCHW."""
    resolved = _resolve_paths(paths)
    required = ("x_test", "y_test")
    _require_files(resolved, required)

    x_test = np.load(resolved["x_test"], mmap_mode="r")
    y_test = np.load(resolved["y_test"], mmap_mode="r")
    if len(x_test) != len(y_test):
        raise ValueError(
            f"x_test e y_test tienen distinto número de muestras: {len(x_test)} != {len(y_test)}"
        )

    sample_index = int(index)
    if sample_index < 0 or sample_index >= len(x_test):
        raise IndexError(
            f"Índice de test fuera de rango: {sample_index}; muestras disponibles={len(x_test)}"
        )

    x_sample = _sample_to_nchw_float32(
        x_test[sample_index:sample_index + 1],
        expected_channels=4,
        name="x_test muestra",
    )
    y_sample = _sample_to_nchw_float32(
        y_test[sample_index:sample_index + 1],
        expected_channels=1,
        name="y_test muestra",
    )
    return x_sample, y_sample
