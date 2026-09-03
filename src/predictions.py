"""Inferencia de una muestra DSM sin modificar el protocolo de entrenamiento."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from .data import DEMO_DATA_PATH, _sample_to_nchw_float32


DEMO_PREDICTION_KEYS = {
    "U-Net": "pred_unet",
    "U-Net++": "pred_unetpp",
    "Attention-U-Net-Residual": "pred_attention",
    "Swin-Tiny-Encoder-CNN-Decoder": "pred_swin",
    "HRNet-W18-Multiscale": "pred_hrnet",
}


def precomputed_demo_available(path: str | Path | None = None) -> bool:
    demo_path = DEMO_DATA_PATH if path is None else Path(path).expanduser().resolve()
    return demo_path.is_file()


def get_precomputed_demo_size(path: str | Path | None = None) -> int:
    demo_path = DEMO_DATA_PATH if path is None else Path(path).expanduser().resolve()
    with np.load(demo_path, allow_pickle=False) as demo:
        required = {"x_test_demo", "y_test_demo", "test_indices"}
        missing = required - set(demo.files)
        if missing:
            raise ValueError("El archivo demo no contiene todos los arrays de test requeridos.")
        sizes = {len(demo[key]) for key in required}
        if len(sizes) != 1:
            raise ValueError("Los arrays de test del archivo demo no están emparejados.")
        sample_count = sizes.pop()
    if sample_count < 1:
        raise ValueError("El subconjunto demo de test está vacío.")
    return sample_count


def load_precomputed_demo_sample(
    model_name: str,
    index: int,
    path: str | Path | None = None,
):
    """Carga entrada, target y predicción precalculada de una muestra demo."""
    if model_name not in DEMO_PREDICTION_KEYS:
        raise ValueError(f"Modelo sin predicción demo asociada: {model_name}")

    demo_path = DEMO_DATA_PATH if path is None else Path(path).expanduser().resolve()
    prediction_key = DEMO_PREDICTION_KEYS[model_name]
    with np.load(demo_path, allow_pickle=False) as demo:
        required = {"x_test_demo", "y_test_demo", "test_indices", prediction_key}
        missing = required - set(demo.files)
        if missing:
            raise ValueError("El archivo demo no contiene la predicción requerida.")
        sizes = {len(demo[key]) for key in required}
        if len(sizes) != 1:
            raise ValueError("Los arrays de predicción demo no están emparejados.")
        sample_count = sizes.pop()
        sample_index = int(index)
        if sample_index < 0 or sample_index >= sample_count:
            raise IndexError("Índice de muestra demo fuera de rango.")

        x_sample = _sample_to_nchw_float32(
            demo["x_test_demo"][sample_index:sample_index + 1],
            expected_channels=4,
            name="x_test demo",
        )
        y_sample = _sample_to_nchw_float32(
            demo["y_test_demo"][sample_index:sample_index + 1],
            expected_channels=1,
            name="y_test demo",
        )
        prediction = _sample_to_nchw_float32(
            demo[prediction_key][sample_index:sample_index + 1],
            expected_channels=1,
            name="predicción demo",
        )
        original_index = int(demo["test_indices"][sample_index])

    return x_sample, y_sample, prediction, original_index

def _extract_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("El checkpoint no contiene un diccionario de pesos válido.")
    for key in ("model", "model_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        return checkpoint
    raise ValueError("No se encontró el estado del modelo dentro del checkpoint.")


def predict_test_sample(model_name: str, x_sample, checkpoint_path: str | Path):
    """Carga un único best checkpoint y devuelve su salida nativa en CPU."""
    import torch

    from .models import create_model

    expected_input = (1, 4, 128, 128)
    expected_output = (1, 1, 128, 128)
    if tuple(x_sample.shape) != expected_input:
        raise ValueError(
            f"La muestra debe tener forma {expected_input}; shape={tuple(x_sample.shape)}"
        )

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("El checkpoint seleccionado no está disponible.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    checkpoint = None
    output = None
    try:
        model = create_model(model_name, use_pretrained=False).to(device)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        state = _extract_model_state(checkpoint)
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            output = model(x_sample.to(device))
        native_shape = tuple(output.shape)
        if native_shape != expected_output:
            raise AssertionError(
                f"Salida nativa {native_shape}; se esperaba exactamente {expected_output}"
            )
        return output.detach().cpu()
    finally:
        if output is not None:
            del output
        if checkpoint is not None:
            del checkpoint
        if model is not None:
            model.to("cpu")
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def sample_error_metrics(prediction, target) -> dict[str, float]:
    """Aplica exactamente las funciones MAE/RMSE ya usadas por el proyecto."""
    from .metrics import mae, rmse

    if tuple(prediction.shape) != (1, 1, 128, 128):
        raise ValueError(f"Forma de predicción inesperada: {tuple(prediction.shape)}")
    if tuple(target.shape) != (1, 1, 128, 128):
        raise ValueError(f"Forma de target inesperada: {tuple(target.shape)}")
    return {
        "MAE": float(mae(prediction, target).item()),
        "RMSE": float(rmse(prediction, target).item()),
    }
