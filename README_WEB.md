# Aplicación web del TFG DSM

Esta interfaz permite consultar resultados existentes y ejecutar entrenamientos reducidos de demostración para las cinco arquitecturas del TFG. No reemplaza el notebook ni el protocolo científico de validación cruzada.

## Instalación

Se recomienda usar un entorno virtual de Python:

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux o macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Por defecto, la aplicación busca:

```text
data/xtrain_nyc-002.npy
data/ytrain_nyc.npy
data/xtest_nyc.npy
data/ytest_nyc.npy
```

Para una demostración web solo se cargan los dos archivos de entrenamiento. Las rutas pueden sobrescribirse sin modificar código:

```powershell
$env:TFG_DATA_DIR = "D:\ruta\a\datos"
$env:TFG_RESULTS_DIR = "D:\ruta\a\resultados"
$env:TFG_CHECKPOINTS_DIR = "D:\ruta\a\checkpoints"
$env:TFG_WEB_RUNS_DIR = "D:\ruta\a\web_runs"
```

En Linux o macOS se usan las mismas variables mediante `export`.

Los datasets y checkpoints están excluidos de Git. No deben subirse al repositorio ni a Streamlit Community Cloud.

## Ejecución

```bash
streamlit run app.py
```

La aplicación no entrena al abrirse. El entrenamiento comienza únicamente después de seleccionar el modo `Entrenar desde la web` y pulsar `Entrenar modelo seleccionado`.

## Modos de uso

### Ver resultados entrenados

La aplicación busca automáticamente:

```text
results/crossval/**/crossval_fold_results.csv
results/crossval/**/crossval_summary.csv
web_runs/**/metrics.csv
```

Los resultados se pueden filtrar por la arquitectura seleccionada. Cuando existen filas fold a fold, el resumen se recalcula agrupando por `run_id` y modelo e informa media y desviación estándar de MAE, RMSE y R².

### Entrenar desde la web

Este modo crea un modelo nuevo, usa únicamente una fracción configurable de `x_train` / `y_train`, genera un split 80/20 de demostración y guarda artefactos aislados:

```text
web_runs/<run_id>/<model_short_name>/
```

Se conservan la loss SmoothL1, AdamW, CosineAnnealingLR, las fórmulas de MAE/RMSE/R² y la selección del mejor checkpoint por menor RMSE de validación.

Para una grabación rápida se recomienda:

- U-Net;
- 1 época;
- 32 muestras;
- batch size 16;
- pesos preentrenados desactivados.

El selector de épocas no impone un máximo. Muchas épocas, el dataset completo, Swin o HRNet pueden tardar mucho o agotar memoria.

## Entrenamiento web frente al protocolo científico

El entrenamiento web sirve para demostrar el flujo técnico y la interfaz. Sus métricas no deben presentarse como resultado final del TFG.

El protocolo científico continúa en `TFG_DsmV6.ipynb`:

- repeated holdout con cinco splits aleatorios 80/20;
- configuración común para los modelos comparados;
- 50 épocas en el experimento completo;
- checkpoints separados;
- evaluación final sobre el test reservado.

La aplicación no modifica ni ejecuta automáticamente ese protocolo.

## Sanity check

El botón `Ejecutar sanity check de arquitecturas` instancia cada modelo sin descargar pesos preentrenados, usa una entrada `(2, 4, 128, 128)` y comprueba la salida nativa exacta `(2, 1, 128, 128)`. Si una arquitectura o dependencia falla, se muestra el error y no se corrige la salida mediante interpolación externa.

## Streamlit Community Cloud

1. Publica únicamente el código y documentación en GitHub.
2. No subas `.npy`, checkpoints, credenciales ni `secrets.toml`.
3. En Streamlit Community Cloud, crea una aplicación apuntando a `app.py`.
4. Configura datos y resultados mediante un almacenamiento permitido y las variables de entorno correspondientes.
5. Verifica que la versión de Python elegida es compatible con PyTorch y timm.

El hosting gratuito suele tener límites estrictos de CPU, RAM, GPU, duración y almacenamiento. Se recomienda desplegar principalmente el modo de consulta y reservar los entrenamientos completos para Colab u otra máquina con GPU.
