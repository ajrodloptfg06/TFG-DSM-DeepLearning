"""Descubrimiento y agregación de resultados existentes sin reentrenar."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CHECKPOINTS_DIR, RESULTS_DIR, WEB_RUNS_DIR


FOLD_RESULT_COLUMNS = [
    "run_id", "model", "fold", "train_size", "val_size", "seed",
    "best_val_RMSE", "best_val_MAE", "best_val_R2", "checkpoint",
    "epochs", "batch_size", "lr", "weight_decay", "source_csv",
]
SUMMARY_COLUMNS = [
    "run_id", "model", "n_folds", "mean_val_RMSE", "std_val_RMSE",
    "mean_val_MAE", "std_val_MAE", "mean_val_R2", "std_val_R2",
]


def _empty_frame(columns: list[str], message: str, warnings=None) -> pd.DataFrame:
    frame = pd.DataFrame(columns=columns)
    frame.attrs["message"] = message
    frame.attrs["warnings"] = list(warnings or [])
    return frame


def _infer_run_id(csv_path: Path) -> str:
    if csv_path.parent.name:
        return csv_path.parent.name
    return "unknown_run"


def _read_csvs(paths: list[Path], *, expected_kind: str) -> tuple[list[pd.DataFrame], list[str]]:
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for csv_path in paths:
        try:
            frame = pd.read_csv(csv_path)
            if "run_id" not in frame.columns:
                frame["run_id"] = _infer_run_id(csv_path)
            frame["source_csv"] = str(csv_path.resolve())
            frames.append(frame)
        except Exception as exc:
            warnings.append(f"No se pudo leer {expected_kind} {csv_path}: {exc}")
    return frames, warnings


def load_all_crossval_results(results_dir: str | Path | None = None) -> pd.DataFrame:
    root = RESULTS_DIR if results_dir is None else Path(results_dir).expanduser().resolve()
    paths = sorted(root.glob("crossval/**/crossval_fold_results.csv"))
    frames, warnings = _read_csvs(paths, expected_kind="resultado por fold")
    if not frames:
        return _empty_frame(
            FOLD_RESULT_COLUMNS,
            "No hay resultados fold a fold disponibles.",
            warnings,
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.attrs["message"] = f"Se cargaron {len(paths)} archivos fold a fold."
    combined.attrs["warnings"] = warnings
    return combined


def load_existing_crossval_summaries(
    results_dir: str | Path | None = None,
) -> pd.DataFrame:
    root = RESULTS_DIR if results_dir is None else Path(results_dir).expanduser().resolve()
    paths = sorted(root.glob("crossval/**/crossval_summary.csv"))
    frames, warnings = _read_csvs(paths, expected_kind="resumen")
    if not frames:
        return _empty_frame(
            SUMMARY_COLUMNS + ["source_csv"],
            "No hay resultados de validación cruzada disponibles en esta versión de la aplicación.",
            warnings,
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.attrs["message"] = f"Se cargaron {len(paths)} resúmenes existentes."
    combined.attrs["warnings"] = warnings
    return combined


def build_crossval_summary(
    fold_results: pd.DataFrame | None = None,
    results_dir: str | Path | None = None,
) -> pd.DataFrame:
    folds = load_all_crossval_results(results_dir) if fold_results is None else fold_results.copy()
    if folds.empty:
        existing = load_existing_crossval_summaries(results_dir)
        if existing.empty:
            message = "No hay resultados de validación cruzada disponibles en esta versión de la aplicación."
            return _empty_frame(SUMMARY_COLUMNS, message, existing.attrs.get("warnings"))
        available = [column for column in SUMMARY_COLUMNS if column in existing.columns]
        summary = existing[available].copy()
        summary.attrs.update(existing.attrs)
        sort_columns = [
            column for column in ("run_id", "mean_val_RMSE")
            if column in summary.columns
        ]
        if sort_columns:
            summary = summary.sort_values(sort_columns)
        return summary.reset_index(drop=True)

    required = {"run_id", "model", "best_val_RMSE", "best_val_MAE", "best_val_R2"}
    missing = required - set(folds.columns)
    if missing:
        return _empty_frame(
            SUMMARY_COLUMNS,
            f"Los resultados fold a fold no contienen las columnas requeridas: {sorted(missing)}",
        )

    metric_columns = ["best_val_RMSE", "best_val_MAE", "best_val_R2"]
    for column in metric_columns:
        folds[column] = pd.to_numeric(folds[column], errors="coerce")

    summary = (
        folds.groupby(["run_id", "model"], dropna=False, sort=False)
        .agg(
            n_folds=("best_val_RMSE", "count"),
            mean_val_RMSE=("best_val_RMSE", "mean"),
            std_val_RMSE=("best_val_RMSE", "std"),
            mean_val_MAE=("best_val_MAE", "mean"),
            std_val_MAE=("best_val_MAE", "std"),
            mean_val_R2=("best_val_R2", "mean"),
            std_val_R2=("best_val_R2", "std"),
        )
        .reset_index()
    )
    summary = summary[SUMMARY_COLUMNS].sort_values(
        ["run_id", "mean_val_RMSE"], ascending=[True, True]
    ).reset_index(drop=True)
    summary.attrs["message"] = "Resumen recalculado desde resultados fold a fold."
    summary.attrs["warnings"] = folds.attrs.get("warnings", [])
    return summary


def load_web_run_results(web_runs_dir: str | Path | None = None) -> pd.DataFrame:
    root = WEB_RUNS_DIR if web_runs_dir is None else Path(web_runs_dir).expanduser().resolve()
    paths = sorted(root.glob("**/metrics.csv"))
    frames, warnings = _read_csvs(paths, expected_kind="resultado web")
    if not frames:
        return _empty_frame(
            [], "Todavía no hay ejecuciones realizadas desde la web.", warnings
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.attrs["message"] = f"Se cargaron {len(paths)} ejecuciones web."
    combined.attrs["warnings"] = warnings
    return combined


def find_best_fold_checkpoint(
    model_name: str,
    results_dir: str | Path | None = None,
    checkpoints_dir: str | Path | None = None,
) -> dict[str, object] | None:
    """Selecciona el fold con menor val RMSE y resuelve su checkpoint local."""
    folds = load_all_crossval_results(results_dir)
    required = {"model", "best_val_RMSE", "checkpoint"}
    if folds.empty or not required.issubset(folds.columns):
        return None

    candidates = folds[folds["model"] == model_name].copy()
    candidates["best_val_RMSE"] = pd.to_numeric(
        candidates["best_val_RMSE"], errors="coerce"
    )
    candidates = candidates.dropna(subset=["best_val_RMSE", "checkpoint"])
    if candidates.empty:
        return None

    row = candidates.sort_values("best_val_RMSE", ascending=True).iloc[0]
    recorded = Path(str(row["checkpoint"])).expanduser()
    checkpoint_root = (
        CHECKPOINTS_DIR
        if checkpoints_dir is None
        else Path(checkpoints_dir).expanduser().resolve()
    )

    possible_paths: list[Path] = [recorded]
    normalized = str(row["checkpoint"]).replace("\\", "/")
    marker = "/checkpoints/"
    if marker in normalized:
        relative_tail = normalized.split(marker, 1)[1]
        possible_paths.append(checkpoint_root / Path(relative_tail))
    elif not recorded.is_absolute():
        possible_paths.append(checkpoint_root / recorded)

    resolved_checkpoint = next(
        (path.resolve() for path in possible_paths if path.is_file()),
        None,
    )
    fold_value = row.get("fold")
    return {
        "checkpoint": resolved_checkpoint,
        "checkpoint_name": recorded.name,
        "fold": None if pd.isna(fold_value) else fold_value,
        "best_val_RMSE": float(row["best_val_RMSE"]),
    }
