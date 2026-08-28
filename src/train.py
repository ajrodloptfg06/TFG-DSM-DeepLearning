"""Entrenamiento reutilizable y wrapper aislado para demostraciones web."""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split

from .config import DEFAULT_SEED
from .data import load_demo_dataset
from .metrics import evaluate_global, train_one_epoch
from .models import MODEL_REGISTRY, create_model
from .utils import get_device, make_run_id, safe_torch_save, seed_everything


ProgressCallback = Callable[[int, int, dict[str, float]], None]


def _checkpoint_compatibility(
    model,
    ckpt_path,
    *,
    run_id,
    seed,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
):
    return {
        "run_id": run_id,
        "run_mode": "web_demo",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "model_class": model.__class__.__name__,
        "checkpoint_path": str(Path(ckpt_path).resolve()),
    }


def _assert_checkpoint_compatible(checkpoint, expected):
    existing = checkpoint.get("compatibility")
    if not isinstance(existing, dict):
        raise RuntimeError("El checkpoint no contiene metadatos de compatibilidad.")
    mismatches = [
        f"{key}: existente={existing.get(key)!r}, actual={value!r}"
        for key, value in expected.items()
        if existing.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Checkpoint incompatible con la ejecución web:\n- " + "\n- ".join(mismatches)
        )


def fit_model_resumable(
    model,
    train_loader,
    val_loader,
    *,
    device,
    epochs,
    learning_rate,
    weight_decay,
    ckpt_path,
    best_ckpt_path,
    run_id,
    seed,
    batch_size,
    allow_resume=False,
    progress_callback: ProgressCallback | None = None,
):
    """Mantiene AdamW, CosineAnnealingLR y menor val_RMSE como criterio best."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    compatibility = _checkpoint_compatibility(
        model,
        ckpt_path,
        run_id=run_id,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    start_epoch = 1
    best_rmse = float("inf")
    history: list[dict[str, float]] = []
    ckpt_path = Path(ckpt_path)
    best_ckpt_path = Path(best_ckpt_path)

    if ckpt_path.exists():
        if not allow_resume:
            raise FileExistsError(
                f"Ya existe {ckpt_path}. Usa otro run_id o activa resume explícitamente."
            )
        checkpoint = torch.load(ckpt_path, map_location=device)
        _assert_checkpoint_compatible(checkpoint, compatibility)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rmse = float(checkpoint.get("best_rmse", best_rmse))
        history = list(checkpoint.get("history", []))

    for epoch in range(start_epoch, epochs + 1):
        started_at = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_global(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_MAE": float(val_metrics["MAE"]),
            "val_RMSE": float(val_metrics["RMSE"]),
            "val_R2": float(val_metrics["R2"]),
            "seconds": float(time.time() - started_at),
        }
        history.append(row)
        is_best = val_metrics["RMSE"] < best_rmse
        if is_best:
            best_rmse = float(val_metrics["RMSE"])

        last_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_rmse": best_rmse,
            "history": history,
            "best_ckpt_path": str(best_ckpt_path),
            "compatibility": compatibility,
        }
        safe_torch_save(last_state, ckpt_path)

        if is_best:
            safe_torch_save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_rmse": best_rmse,
                    "history": history,
                    "compatibility": {
                        **compatibility,
                        "checkpoint_path": str(best_ckpt_path.resolve()),
                    },
                },
                best_ckpt_path,
            )

        if progress_callback is not None:
            progress_callback(epoch, epochs, dict(row))

    return history, best_rmse


def run_web_training(
    model_name: str,
    epochs: int,
    max_samples: int | None = 128,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    use_pretrained: bool = False,
    output_dir: str = "web_runs",
    *,
    progress_callback: ProgressCallback | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Ejecuta una demostración aislada; no sustituye al protocolo científico."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo desconocido {model_name!r}.")
    if int(epochs) < 1:
        raise ValueError("epochs debe ser al menos 1.")
    if int(batch_size) < 1:
        raise ValueError("batch_size debe ser al menos 1.")
    if float(learning_rate) <= 0 or float(weight_decay) < 0:
        raise ValueError("learning_rate debe ser positivo y weight_decay no negativo.")

    epochs = int(epochs)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    seed_everything(seed)
    device = get_device()

    x_data, y_data = load_demo_dataset(max_samples=max_samples)
    dataset = TensorDataset(x_data, y_data)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("El split demo necesita al menos una muestra de entrenamiento.")
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    run_id = make_run_id("web")
    spec = MODEL_REGISTRY[model_name]
    run_dir = Path(output_dir).expanduser().resolve() / run_id / spec["short_name"]
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_last = run_dir / f"last_{spec['short_name']}.pth"
    checkpoint_best = run_dir / f"best_{spec['short_name']}.pth"

    model = None
    history = None
    best_checkpoint = None
    try:
        model = create_model(model_name, use_pretrained=use_pretrained)
        history, _ = fit_model_resumable(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            ckpt_path=checkpoint_last,
            best_ckpt_path=checkpoint_best,
            run_id=run_id,
            seed=seed,
            batch_size=batch_size,
            allow_resume=False,
            progress_callback=progress_callback,
        )
        best_checkpoint = torch.load(checkpoint_best, map_location=device)
        model.load_state_dict(best_checkpoint["model"])
        final_metrics = evaluate_global(model, val_loader, device)
        best_row = min(history, key=lambda row: float(row["val_RMSE"]))

        result = {
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_loss": float(best_row["train_loss"]),
            "val_loss": float(final_metrics["loss"]),
            "val_MAE": float(final_metrics["MAE"]),
            "val_RMSE": float(final_metrics["RMSE"]),
            "val_R2": float(final_metrics["R2"]),
            "checkpoint_best": str(checkpoint_best),
            "checkpoint_last": str(checkpoint_last),
            "run_id": run_id,
        }
        pd.DataFrame([result]).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        return result
    finally:
        if best_checkpoint is not None:
            del best_checkpoint
        if model is not None:
            model.to("cpu")
            del model
        if history is not None:
            del history
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
