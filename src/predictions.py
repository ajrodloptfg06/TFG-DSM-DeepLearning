"""Inferencia de una muestra DSM sin modificar el protocolo de entrenamiento."""

from __future__ import annotations

import gc
from pathlib import Path


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
