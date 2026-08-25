# Validación repetida 80/20

## Objetivo

El notebook principal incorpora un pipeline opcional de *repeated holdout* para estimar la estabilidad de las métricas de validación frente a distintas particiones de los datos de entrenamiento.

Se generan cinco particiones aleatorias reproducibles. En cada una se usa el 80 % de `x_train` / `y_train` para entrenamiento y el 20 % restante para validación. El conjunto de test queda reservado para la evaluación final oficial y no interviene en este pipeline.

## No es un 5-fold clásico

Las cinco validaciones no forman folds disjuntos que cubran el dataset exactamente una vez. Cada split se genera mediante una permutación independiente con semilla `CV_SEED + fold_idx`; por tanto, una muestra puede pertenecer a validación en más de un split o en ninguno.

Esta elección implementa literalmente cinco particiones aleatorias 80/20. Su propósito es medir sensibilidad al split, no sustituir una validación cruzada K-fold clásica.

## Configuración segura

La configuración por defecto es:

```python
RUN_CROSS_VALIDATION = False
CV_N_SPLITS = 5
CV_TRAIN_RATIO = 0.8
CV_SEED = SEED
CV_MODEL_NAMES = ["Attention-U-Net-Residual"]
```

El pipeline no inicia ningún entrenamiento mientras `RUN_CROSS_VALIDATION` sea `False`. Para la primera ejecución se recomienda conservar únicamente Attention U-Net residual. Se puede ampliar `CV_MODEL_NAMES` con las etiquetas exactas del registro:

- `U-Net`
- `U-Net++`
- `Attention-U-Net-Residual`
- `Swin-Tiny-Encoder-CNN-Decoder`
- `HRNet-W18-Multiscale`

Ejecutar los cinco modelos con cinco splits supone 25 entrenamientos y puede ser costoso en Colab.

## Reproducibilidad de los splits

Cada fold registra:

- identificador y semilla efectiva;
- tamaños de entrenamiento y validación;
- hashes SHA-256 de los índices ordenados de cada subconjunto;
- tamaño total del dataset y algoritmo de generación implícito en el código.

Los índices se regeneran determinísticamente con `torch.randperm` y un `torch.Generator` inicializado con `CV_SEED + fold_idx`. El manifiesto permite comprobar que una reanudación conserva exactamente la configuración y los hashes.

## Aislamiento de artefactos

Los artefactos no comparten carpetas con los runs FAST o FINAL:

```text
checkpoints/crossval/<RUN_ID>/<model_slug>/fold_XX/
results/crossval/<RUN_ID>/
```

Cada fold guarda `last_<model>.pth` y `best_<model>.pth`. Los resultados agregados se escriben como:

- `crossval_fold_results.csv`: una fila por modelo y split;
- `crossval_summary.csv`: media y desviación estándar por modelo;
- `crossval_manifest.json`: configuración, splits, versiones, estado y progreso.

Si ya existen artefactos para el mismo `RUN_ID`, el pipeline falla con `ALLOW_RESUME=False`. La reanudación requiere `TFG_ALLOW_RESUME=true` y coincidencia exacta con el manifiesto CV existente.

## Entrenamiento y métricas

Cada fold crea nuevos `Subset`, `DataLoader` y modelo. `fit_model_resumable` crea también un AdamW y un CosineAnnealingLR nuevos para ese modelo y fold. Se mantienen sin cambios la loss, las métricas, el optimizador, el scheduler, el batch size y el número de épocas del protocolo principal.

El mejor checkpoint se selecciona por `val_RMSE`. De la misma época se registran `best_val_MAE` y `best_val_R2`. Al terminar cada fold, el modelo se mueve a CPU, se eliminan referencias, se ejecuta `gc.collect()` y se vacía la caché CUDA cuando está disponible.

La tabla agregada informa media y desviación estándar muestral (`ddof=1`) de RMSE, MAE y R². El formato `media ± desviación` expresa estabilidad entre particiones; no demuestra por sí solo superioridad estadística.

## Procedimiento recomendado en Colab

1. Ejecutar el notebook hasta completar carga de datos, definiciones y sanity check.
2. Mantener desactivados los entrenamientos FINAL si solo se va a ejecutar validación repetida.
3. Elegir un `RUN_ID` nuevo.
4. Dejar inicialmente `CV_MODEL_NAMES=["Attention-U-Net-Residual"]`.
5. Cambiar `RUN_CROSS_VALIDATION=True` y volver a ejecutar desde la configuración CV.
6. Comprobar las rutas impresas antes de iniciar.
7. Ejecutar la celda de control.
8. Revisar `crossval_fold_results.csv`, `crossval_summary.csv` y `crossval_manifest.json`.
9. Ampliar la lista de modelos únicamente si existen tiempo, GPU y espacio suficientes.

## Relación con el resultado final

La comparación principal del TFG sigue siendo el run FINAL: los cinco mejores checkpoints se evalúan una única vez sobre el mismo conjunto de test reservado. La validación repetida añade evidencia sobre estabilidad en distintas separaciones train/validation, pero no reemplaza ni se combina con las métricas de test.
