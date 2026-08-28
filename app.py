"""Interfaz Streamlit para demostraciones y consulta de resultados DSM."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import RESULTS_DIR, WEB_RUNS_DIR
from src.results import (
    build_crossval_summary,
    load_all_crossval_results,
    load_web_run_results,
)


MODEL_NAMES = (
    "U-Net",
    "U-Net++",
    "Attention-U-Net-Residual",
    "Swin-Tiny-Encoder-CNN-Decoder",
    "HRNet-W18-Multiscale",
)


def _show_dataframe_message(frame: pd.DataFrame) -> None:
    message = frame.attrs.get("message")
    if message:
        st.info(message)
    for warning in frame.attrs.get("warnings", []):
        st.warning(warning)


def _render_results(model_name: str) -> None:
    st.subheader("Resultados de validación cruzada")
    folds = load_all_crossval_results(RESULTS_DIR)
    summary = build_crossval_summary(folds, RESULTS_DIR)

    if not summary.empty and "model" in summary.columns:
        filtered_summary = summary[summary["model"] == model_name]
        if filtered_summary.empty:
            st.info(f"No hay resumen de validación cruzada para {model_name}.")
        else:
            st.dataframe(filtered_summary, use_container_width=True, hide_index=True)
    else:
        _show_dataframe_message(summary)

    st.subheader("Resultados fold a fold")
    if not folds.empty and "model" in folds.columns:
        filtered_folds = folds[folds["model"] == model_name]
        if filtered_folds.empty:
            st.info(f"No hay filas fold a fold para {model_name}.")
        else:
            st.dataframe(filtered_folds, use_container_width=True, hide_index=True)
    else:
        _show_dataframe_message(folds)

    st.subheader("Ejecuciones web anteriores")
    web_results = load_web_run_results(WEB_RUNS_DIR)
    if not web_results.empty and "model" in web_results.columns:
        filtered_web = web_results[web_results["model"] == model_name]
        if filtered_web.empty:
            st.info(f"No hay ejecuciones web guardadas para {model_name}.")
        else:
            st.dataframe(filtered_web, use_container_width=True, hide_index=True)
    else:
        _show_dataframe_message(web_results)


def _render_sanity_check() -> None:
    st.divider()
    st.subheader("Sanity check de arquitecturas")
    st.caption(
        "Instancia cada arquitectura sin pesos preentrenados y valida su salida nativa."
    )
    if st.button("Ejecutar sanity check de arquitecturas"):
        try:
            from src.models import sanity_check_models

            with st.spinner("Comprobando las cinco arquitecturas..."):
                report = sanity_check_models(device="cpu")
            st.dataframe(pd.DataFrame(report), use_container_width=True, hide_index=True)
            if all(row["ok"] for row in report):
                st.success("Todas las arquitecturas devuelven (2, 1, 128, 128).")
            else:
                st.error(
                    "Alguna arquitectura no superó el sanity check. Revisa la columna error; "
                    "la aplicación no modifica el modelo para corregirlo."
                )
        except ImportError as exc:
            st.error(
                "Faltan dependencias de deep learning. Ejecuta "
                f"`pip install -r requirements.txt`. Detalle: {exc}"
            )
        except Exception as exc:
            st.error(f"No se pudo completar el sanity check: {type(exc).__name__}: {exc}")


def _render_training(
    model_name: str,
    epochs: int,
    batch_size: int,
    max_samples: int | None,
    learning_rate: float,
    use_pretrained: bool,
) -> None:
    st.warning(
        "El entrenamiento desde la web está pensado como demostración. "
        "Los experimentos científicos completos se realizan mediante validación "
        "cruzada de 5 folds y 50 épocas."
    )
    st.write(
        "El entrenamiento solo comienza al pulsar el botón. Para grabar una demo, "
        "se recomienda 1 época y 32 muestras."
    )

    if not st.button("Entrenar modelo seleccionado", type="primary"):
        return

    progress_bar = st.progress(0.0)
    status = st.empty()

    def update_progress(epoch: int, total_epochs: int, row: dict[str, float]) -> None:
        progress_bar.progress(epoch / total_epochs)
        status.write(
            f"Época {epoch}/{total_epochs} · "
            f"loss={row['val_loss']:.4f} · RMSE={row['val_RMSE']:.4f}"
        )

    try:
        from src.train import run_web_training

        with st.spinner("Entrenando el modelo seleccionado..."):
            result = run_web_training(
                model_name=model_name,
                epochs=epochs,
                max_samples=max_samples,
                batch_size=batch_size,
                learning_rate=learning_rate,
                use_pretrained=use_pretrained,
                output_dir=str(WEB_RUNS_DIR),
                progress_callback=update_progress,
            )
        progress_bar.progress(1.0)
        status.success("Entrenamiento de demostración terminado.")
        metrics = pd.DataFrame([{
            "loss": result["val_loss"],
            "MAE": result["val_MAE"],
            "RMSE": result["val_RMSE"],
            "R2": result["val_R2"],
        }])
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.write(f"Run ID: `{result['run_id']}`")
        st.write(f"Checkpoint best: `{result['checkpoint_best']}`")
        st.write(f"Checkpoint last: `{result['checkpoint_last']}`")
    except FileNotFoundError as exc:
        st.error(str(exc))
    except ImportError as exc:
        st.error(
            "Faltan dependencias de deep learning. Ejecuta "
            f"`pip install -r requirements.txt`. Detalle: {exc}"
        )
    except Exception as exc:
        st.error(f"El entrenamiento no pudo completarse: {type(exc).__name__}: {exc}")


def main() -> None:
    st.set_page_config(page_title="Estimación de DSM", layout="wide")
    st.title("Estimación de DSM mediante Deep Learning")
    st.write(
        "Aplicación web básica para seleccionar arquitecturas, lanzar entrenamientos "
        "reducidos o configurables y visualizar métricas de evaluación."
    )

    st.sidebar.header("Configuración")
    model_name = st.sidebar.selectbox("Arquitectura", MODEL_NAMES)
    mode = st.sidebar.radio(
        "Modo",
        ("Ver resultados entrenados", "Entrenar desde la web"),
    )
    epochs = int(st.sidebar.number_input("Número de épocas", min_value=1, value=1, step=1))
    batch_size = int(st.sidebar.number_input("Batch size", min_value=1, value=16, step=1))
    use_full_dataset = st.sidebar.checkbox("Usar todo el train en la demo", value=False)
    max_samples_value = int(
        st.sidebar.number_input("Máximo de muestras demo", min_value=2, value=32, step=1)
    )
    max_samples = None if use_full_dataset else max_samples_value
    learning_rate = float(
        st.sidebar.number_input(
            "Learning rate",
            min_value=1e-8,
            value=1e-4,
            format="%.8f",
        )
    )
    use_pretrained = st.sidebar.checkbox("Usar backbone preentrenado", value=False)
    if epochs > 5 or max_samples is None:
        st.sidebar.warning(
            "Esta configuración puede tardar mucho o superar los recursos de un hosting gratuito."
        )

    if mode == "Ver resultados entrenados":
        _render_results(model_name)
    else:
        _render_training(
            model_name,
            epochs,
            batch_size,
            max_samples,
            learning_rate,
            use_pretrained,
        )

    _render_sanity_check()


if __name__ == "__main__":
    main()
