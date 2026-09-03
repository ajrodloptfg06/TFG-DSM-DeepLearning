# Aplicación web del TFG DSM

Esta interfaz protegida permite consultar resultados existentes, generar una predicción DSM individual y ejecutar entrenamientos reducidos de demostración para las cinco arquitecturas del TFG. No reemplaza el notebook ni el protocolo científico de validación cruzada.

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

### Configurar el acceso

La aplicación exige `APP_USERNAME` y `APP_PASSWORD`. No hay credenciales incluidas en el código. Para una ejecución local en PowerShell, usa valores propios; los siguientes son únicamente ficticios:

```powershell
$env:APP_USERNAME = "usuario_demo"
$env:APP_PASSWORD = "cambia_esta_clave_ficticia"
streamlit run app.py
```

En Linux o macOS:

```bash
export APP_USERNAME="usuario_demo"
export APP_PASSWORD="cambia_esta_clave_ficticia"
streamlit run app.py
```

En Streamlit Community Cloud, abre la configuración de la aplicación, entra en `Secrets` y añade valores propios:

```toml
APP_USERNAME = "usuario_demo"
APP_PASSWORD = "cambia_esta_clave_ficticia"
```

No subas `.streamlit/secrets.toml` al repositorio ni imprimas las credenciales en logs. La sesión permanece autenticada durante la navegación y puede cerrarse desde `Cerrar sesión` en la barra lateral.

### Arrancar la aplicación

```bash
streamlit run app.py
```

La aplicación no entrena al abrirse. El entrenamiento comienza únicamente después de seleccionar el modo `Entrenar desde la web` y pulsar `Entrenar modelo seleccionado`.

## Modos de uso

Tras iniciar sesión, la navegación ofrece `Resultados`, `Predicciones DSM` y `Entrenamiento demo`.

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

### Predicciones DSM

La página carga mediante `mmap` únicamente la muestra de test seleccionada desde `xtest_nyc.npy` y `ytest_nyc.npy`. Admite muestras guardadas como NHWC o NCHW y valida las formas `(1, 4, 128, 128)` y `(1, 1, 128, 128)` después de la conversión. `TFG_DATA_DIR` permite configurar la carpeta de datos sin cambiar el código.

Los cuatro canales se normalizan min-max solo para mostrarlos en pantalla. El tensor original, sin esa transformación visual, es el que recibe el modelo.

Para cada arquitectura, la aplicación consulta los CSV fold a fold y selecciona un único checkpoint: el del split con menor `best_val_RMSE`. No realiza ensemble. La resolución local conserva la ruta registrada cuando existe o reconstruye su ubicación bajo `TFG_CHECKPOINTS_DIR`:

```powershell
$env:TFG_CHECKPOINTS_DIR = "D:\ruta\a\checkpoints"
```

Al pulsar `Generar predicción`, se instancia solamente el modelo elegido, se cargan sus pesos, se exige una salida nativa exacta `(1, 1, 128, 128)` sin interpolación y se libera el modelo al terminar. La página muestra ground truth, predicción, error absoluto y MAE/RMSE de esa muestra usando las funciones existentes del proyecto. No entrena ni escribe archivos.

Los datasets y checkpoints están excluidos de Git, por lo que normalmente no estarán disponibles en el despliegue público. En ese caso la aplicación muestra un aviso limpio y sigue permitiendo consultar los resultados CSV versionados.

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
4. Configura `APP_USERNAME` y `APP_PASSWORD` en `Secrets` con valores privados.
5. Configura datos, checkpoints y resultados mediante un almacenamiento permitido y las variables de entorno correspondientes si deseas habilitar predicciones.
6. Verifica que la versión de Python elegida es compatible con PyTorch y timm.

El hosting gratuito suele tener límites estrictos de CPU, RAM, GPU, duración y almacenamiento. Se recomienda desplegar principalmente el modo de consulta y reservar los entrenamientos completos para Colab u otra máquina con GPU.
