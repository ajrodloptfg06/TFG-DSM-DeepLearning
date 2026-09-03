"""Interfaz Streamlit para demostraciones y consulta de resultados DSM."""

from __future__ import annotations

import hmac
import os
import re
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import streamlit as st

from src.config import RESULTS_DIR, WEB_RUNS_DIR
from src.results import (
    build_crossval_summary,
    find_best_fold_checkpoint,
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


def _configured_credential(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    if value is None:
        value = os.environ.get(name)
    return str(value) if value else None


def _render_login() -> bool:
    st.title("Acceso a la aplicación DSM")
    st.write("Identifícate para consultar resultados, predicciones y demostraciones.")

    expected_username = _configured_credential("APP_USERNAME")
    expected_password = _configured_credential("APP_PASSWORD")
    missing = [
        name
        for name, value in (
            ("APP_USERNAME", expected_username),
            ("APP_PASSWORD", expected_password),
        )
        if value is None
    ]
    if missing:
        st.error("La autenticación de la aplicación no está configurada.")
        _render_technical_details([
            "Variables de configuración ausentes: " + ", ".join(missing)
        ])
        return False

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Iniciar sesión", type="primary")

    if submitted:
        valid_username = hmac.compare_digest(username, expected_username)
        valid_password = hmac.compare_digest(password, expected_password)
        if valid_username and valid_password:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Usuario o contraseña incorrectos.")
    return False


def _render_technical_details(details: list[str] | tuple[str, ...]) -> None:
    clean_details = []
    internal_path = re.compile(
        r"(?i)(?:[A-Z]:[\\/]|(?:^|[\s\"'(=:])/(?!/)[^\s]*)"
    )
    for detail in details:
        if not detail:
            continue
        text = str(detail)
        if internal_path.search(text):
            error_type = text.split(":", 1)[0]
            text = f"{error_type}: información de ruta interna omitida."
        clean_details.append(text)
    if not clean_details:
        return
    with st.expander("Detalles técnicos", expanded=False):
        for detail in clean_details:
            st.code(detail, language=None)


def _show_dataframe_message(frame: pd.DataFrame) -> None:
    message = frame.attrs.get("message")
    if message:
        st.info(message)
    _render_technical_details(frame.attrs.get("warnings", []))


def _basename_only(value):
    if pd.isna(value):
        return value
    normalized = str(value).replace("\\", "/")
    return PurePosixPath(normalized).name


def _presentable_value(value):
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        return PurePosixPath(normalized).name
    return value


def _presentable_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Elimina rutas internas de las tablas destinadas a la presentación."""
    visible = frame.copy()
    visible = visible.drop(columns=["source_csv"], errors="ignore")
    for column in ("checkpoint", "checkpoint_best", "checkpoint_last"):
        if column in visible.columns:
            visible[column] = visible[column].map(_basename_only)
    for column in visible.select_dtypes(include=["object"]).columns:
        visible[column] = visible[column].map(_presentable_value)
    return visible.rename(columns={"fold": "partición"})


def _render_results(model_name: str) -> None:
    st.subheader("Resultados de validación repetida")
    st.caption("5 particiones aleatorias independientes con split 80/20.")
    folds = load_all_crossval_results(RESULTS_DIR)
    summary = build_crossval_summary(folds, RESULTS_DIR)

    if not summary.empty and "model" in summary.columns:
        filtered_summary = summary[summary["model"] == model_name]
        if filtered_summary.empty:
            st.info(f"No hay resumen de validación repetida para {model_name}.")
        else:
            st.dataframe(
                _presentable_table(filtered_summary),
                use_container_width=True,
                hide_index=True,
            )
    else:
        _show_dataframe_message(summary)

    st.subheader("Resultados por partición")
    if not folds.empty and "model" in folds.columns:
        filtered_folds = folds[folds["model"] == model_name]
        if filtered_folds.empty:
            st.info(f"No hay resultados por partición para {model_name}.")
        else:
            st.dataframe(
                _presentable_table(filtered_folds),
                use_container_width=True,
                hide_index=True,
            )
    else:
        _show_dataframe_message(folds)

    st.subheader("Ejecuciones web anteriores")
    web_results = load_web_run_results(WEB_RUNS_DIR)
    if not web_results.empty and "model" in web_results.columns:
        filtered_web = web_results[web_results["model"] == model_name]
        if filtered_web.empty:
            st.info(f"No hay ejecuciones web guardadas para {model_name}.")
        else:
            st.dataframe(
                _presentable_table(filtered_web),
                use_container_width=True,
                hide_index=True,
            )
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
            report_frame = pd.DataFrame(report)
            technical_errors = [
                f"{row['model']}: {row['error']}"
                for row in report
                if row.get("error")
            ]
            if "error" in report_frame.columns:
                report_frame.loc[report_frame["error"].notna(), "error"] = (
                    "Error detectado (consulta los detalles técnicos)"
                )
            st.dataframe(report_frame, use_container_width=True, hide_index=True)
            _render_technical_details(technical_errors)
            if all(row["ok"] for row in report):
                st.success("Todas las arquitecturas devuelven (2, 1, 128, 128).")
            else:
                st.error(
                    "Alguna arquitectura no superó el sanity check. Revisa la columna error; "
                    "la aplicación no modifica el modelo para corregirlo."
                )
        except ImportError as exc:
            st.error(
                "Faltan dependencias de deep learning para ejecutar esta comprobación."
            )
            _render_technical_details([f"{type(exc).__name__}: {exc}"])
        except Exception as exc:
            st.error("No se pudo completar el sanity check de arquitecturas.")
            _render_technical_details([f"{type(exc).__name__}: {exc}"])


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
        "Los experimentos científicos completos usan validación repetida con "
        "5 particiones aleatorias independientes 80/20 y 50 épocas."
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
        st.write(f"Checkpoint best: `{_basename_only(result['checkpoint_best'])}`")
        st.write(f"Checkpoint last: `{_basename_only(result['checkpoint_last'])}`")
    except FileNotFoundError as exc:
        st.error(
            "No se encontraron los datos necesarios para iniciar el entrenamiento de demostración."
        )
        _render_technical_details([f"{type(exc).__name__}: {exc}"])
    except ImportError as exc:
        st.error(
            "Faltan dependencias de deep learning para iniciar el entrenamiento."
        )
        _render_technical_details([f"{type(exc).__name__}: {exc}"])
    except Exception as exc:
        st.error("El entrenamiento de demostración no pudo completarse.")
        _render_technical_details([f"{type(exc).__name__}: {exc}"])


def _array_stats(values: np.ndarray) -> str:
    values = np.asarray(values)
    return (
        f"mín={np.nanmin(values):.4f} · máx={np.nanmax(values):.4f} · "
        f"media={np.nanmean(values):.4f}"
    )


def _normalized_for_display(values: np.ndarray) -> np.ndarray:
    """Normalización visual independiente del tensor enviado al modelo."""
    values = np.asarray(values, dtype=np.float32)
    minimum = np.nanmin(values)
    maximum = np.nanmax(values)
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)


def _render_scalar_field(values: np.ndarray, title: str, cmap: str = "viridis") -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(values, cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def _render_sample_visuals(x_sample, y_sample) -> np.ndarray:
    st.markdown("#### Canales de entrada")
    channel_columns = st.columns(4)
    for channel_index, column in enumerate(channel_columns):
        original = x_sample[0, channel_index].detach().cpu().numpy()
        with column:
            st.image(
                _normalized_for_display(original),
                caption=f"Canal {channel_index + 1}",
                clamp=True,
                use_container_width=True,
            )
            st.caption(_array_stats(original))

    target = y_sample[0, 0].detach().cpu().numpy()
    st.markdown("#### DSM real")
    target_column, target_stats_column = st.columns([2, 1])
    with target_column:
        _render_scalar_field(target, "DSM real")
    with target_stats_column:
        st.metric("Mínimo", f"{np.nanmin(target):.4f}")
        st.metric("Máximo", f"{np.nanmax(target):.4f}")
        st.metric("Media", f"{np.nanmean(target):.4f}")
    return target


def _render_prediction_visuals(prediction_tensor, y_sample, target: np.ndarray) -> None:
    from src.predictions import sample_error_metrics

    metrics = sample_error_metrics(prediction_tensor, y_sample)
    prediction = prediction_tensor[0, 0].detach().cpu().numpy()
    absolute_error = np.abs(prediction - target)
    prediction_column, error_column = st.columns(2)
    with prediction_column:
        _render_scalar_field(prediction, "DSM predicho")
    with error_column:
        _render_scalar_field(absolute_error, "Error absoluto", cmap="magma")

    mae_column, rmse_column = st.columns(2)
    mae_column.metric("MAE de la muestra", f"{metrics['MAE']:.6f}")
    rmse_column.metric("RMSE de la muestra", f"{metrics['RMSE']:.6f}")


def _render_predictions() -> None:
    st.subheader("Predicciones DSM")
    st.write(
        "Selecciona una arquitectura y una muestra reservada de test para visualizar "
        "la entrada, el DSM real y una inferencia individual."
    )
    model_name = st.selectbox(
        "Arquitectura para la predicción",
        MODEL_NAMES,
        key="prediction_model",
    )

    from src.predictions import precomputed_demo_available

    if precomputed_demo_available():
        try:
            from src.predictions import (
                get_precomputed_demo_size,
                load_precomputed_demo_sample,
            )

            demo_size = get_precomputed_demo_size()
            sample_index = int(
                st.number_input(
                    "Muestra demo",
                    min_value=0,
                    max_value=demo_size - 1,
                    value=0,
                    step=1,
                )
            )
            x_sample, y_sample, prediction_tensor, original_index = (
                load_precomputed_demo_sample(model_name, sample_index)
            )
            st.caption(
                f"Muestra demo {sample_index} "
                f"(índice original de test: {original_index})"
            )
            target = _render_sample_visuals(x_sample, y_sample)
            _render_prediction_visuals(prediction_tensor, y_sample, target)
            st.caption(
                "Predicción precalculada con el checkpoint científico del run FINAL. "
                "Las métricas mostradas corresponden únicamente a la muestra visualizada."
            )
        except (FileNotFoundError, ValueError, IndexError) as exc:
            st.info("Los datos demo de predicción no están disponibles correctamente.")
            _render_technical_details([f"{type(exc).__name__}: {exc}"])
        except ImportError as exc:
            st.error("Faltan dependencias para mostrar la predicción demo.")
            _render_technical_details([f"{type(exc).__name__}: {exc}"])
        return

    try:
        from src.data import get_test_dataset_size, load_test_sample

        test_size = get_test_dataset_size()
        sample_index = int(
            st.number_input(
                "Índice de muestra de test",
                min_value=0,
                max_value=test_size - 1,
                value=0,
                step=1,
            )
        )
        x_sample, y_sample = load_test_sample(sample_index)
    except (FileNotFoundError, ValueError, IndexError) as exc:
        st.info("Los datos de test no están disponibles en esta versión desplegada.")
        _render_technical_details([f"{type(exc).__name__}: {exc}"])
        return
    except ImportError as exc:
        st.error("Faltan dependencias para cargar los datos de test.")
        _render_technical_details([f"{type(exc).__name__}: {exc}"])
        return

    target = _render_sample_visuals(x_sample, y_sample)

    checkpoint_info = find_best_fold_checkpoint(model_name)
    checkpoint_path = None if checkpoint_info is None else checkpoint_info["checkpoint"]
    if checkpoint_path is None:
        st.info("No hay checkpoint disponible para este modelo en esta versión desplegada.")
    else:
        st.caption(
            "Se usará el checkpoint del mejor split según RMSE de validación; "
            "no se realiza ensemble."
        )
        fold = checkpoint_info.get("fold")
        fold_label = "no disponible" if fold is None else str(fold)
        st.write(
            f"Split seleccionado: `{fold_label}` · "
            f"RMSE de validación: `{checkpoint_info['best_val_RMSE']:.6f}`"
        )

    generate = st.button(
        "Generar predicción",
        type="primary",
        disabled=checkpoint_path is None,
    )
    if not generate:
        return

    try:
        from src.predictions import predict_test_sample

        with st.spinner("Generando predicción con el checkpoint seleccionado..."):
            prediction_tensor = predict_test_sample(model_name, x_sample, checkpoint_path)
        _render_prediction_visuals(prediction_tensor, y_sample, target)
        st.caption(
            "Estas métricas corresponden únicamente a la muestra visualizada y no "
            "sustituyen la evaluación científica completa."
        )
    except ImportError as exc:
        st.error("Faltan dependencias de deep learning para generar la predicción.")
        _render_technical_details([f"{type(exc).__name__}: {exc}"])
    except Exception as exc:
        st.error("No se pudo generar la predicción con el checkpoint seleccionado.")
        _render_technical_details([f"{type(exc).__name__}: {exc}"])


def main() -> None:
    st.set_page_config(page_title="Estimación de DSM", layout="wide")

    if not st.session_state.get("authenticated", False):
        _render_login()
        return

    st.sidebar.success("Sesión iniciada")
    if st.sidebar.button("Cerrar sesión"):
        st.session_state.pop("authenticated", None)
        st.rerun()

    st.title("Estimación de DSM mediante Deep Learning")
    st.write(
        "Aplicación web básica para seleccionar arquitecturas, lanzar entrenamientos "
        "reducidos o configurables y visualizar métricas de evaluación."
    )
    st.info(
        "Los resultados científicos completos proceden de los entrenamientos de "
        "validación repetida con 5 particiones aleatorias independientes 80/20. "
        "El entrenamiento desde la web está pensado como demostración."
    )

    st.sidebar.header("Navegación")
    page = st.sidebar.radio(
        "Sección",
        ("Resultados", "Predicciones DSM", "Entrenamiento demo"),
    )

    if page == "Resultados":
        model_name = st.sidebar.selectbox("Arquitectura", MODEL_NAMES)
        _render_results(model_name)
    elif page == "Predicciones DSM":
        _render_predictions()
    else:
        model_name = st.sidebar.selectbox("Arquitectura", MODEL_NAMES)
        demo_training_info = None
        try:
            from src.data import get_demo_training_info

            demo_training_info = get_demo_training_info()
        except (FileNotFoundError, ValueError, ImportError):
            pass
        epochs = int(
            st.sidebar.number_input("Número de épocas", min_value=1, value=1, step=1)
        )
        batch_size = int(
            st.sidebar.number_input("Batch size", min_value=1, value=16, step=1)
        )
        if demo_training_info and demo_training_info["source"] == "demo":
            st.sidebar.caption(
                f"Subconjunto web disponible: {demo_training_info['sample_count']} muestras."
            )
        use_full_dataset = st.sidebar.checkbox(
            "Usar todas las muestras disponibles en la demo", value=False
        )
        available_samples = (
            int(demo_training_info["sample_count"])
            if demo_training_info
            else 32
        )
        max_samples_value = int(
            st.sidebar.number_input(
                "Máximo de muestras demo",
                min_value=2,
                max_value=available_samples,
                value=min(32, available_samples),
                step=1,
            )
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
