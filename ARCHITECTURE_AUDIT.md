# Auditoria tecnica de `TFG_DsmV6.ipynb`

Repositorio: `ajrodloptfg06/TFG-DSM-DeepLearning`

Notebook auditado: `TFG_DsmV6.ipynb`

Fecha de auditoria: 2026-06-09

Referencia externa usada para contexto: el articulo `Derivation of surface models using satellite imagery deep learning architectures with explainable AI` compara DeepLabv3+, SegNet, U-Net y U-Net++ para generar DSM en New York City con imagenes radar, visible/NIR y DSM LiDAR, y reporta MAE, RMSE y R2. Fuente: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2590123024016888), [DOAJ](https://doaj.org/article/a9dc6f8d95cb45d8b5526bb9ae44fb95).

## Resumen ejecutivo

La auditoria apunta a que las diferencias con el paper pueden venir mas de problemas de pipeline, trazabilidad y comparabilidad experimental que de una inferioridad real de las nuevas arquitecturas.

Los modelos principales aceptan la entrada esperada `(batch, 4, 128, 128)` y devuelven o se han probado devolviendo `(batch, 1, 128, 128)`. Sin embargo, hay problemas de gravedad alta: los checkpoints guardados no preservan necesariamente el mejor modelo, U-Net y U-Net++ se entrenan con un nombre de checkpoint pero se intentan cargar desde otro, U-Net++ se evalua llamando a una funcion `evaluate` que no esta definida en el notebook actual, y los resultados guardados mezclan ejecuciones antiguas con codigo nuevo.

Tambien hay incertidumbre importante en el preprocesado: no hay normalizacion explicita por canal dentro del notebook, no se documenta si `xtrain_nyc-002.npy` ya esta normalizado, no se normaliza ni escala el DSM objetivo, y no existe una funcion de desnormalizacion antes de MAE/RMSE/R2. Si el paper calcula metricas en metros y tus `y_train/y_test` estan en metros, esto puede ser correcto; si el paper entreno/evaluo con otro escalado, las metricas no son comparables.

## Tabla de formas por modelo

| Modelo | Definicion/uso | Entrada `(B,4,128,128)` | Salida `(B,1,128,128)` | Observacion |
|---|---:|---:|---:|---|
| Swin-UNet | celdas 12-15 | Si | Si, probado en salida guardada | Usa `timm` con `features_only=True`, `img_size=128`, `in_chans=4`; interpola la salida final a la resolucion original. |
| Attention U-Net | celdas 18-21 | Si | Si, probado en salida guardada | Arquitectura encoder-decoder completa con 4 bajadas y salida 1 canal. |
| HRNet | celdas 24-26 | Si | Si, probado en salida guardada | Usa ultimo feature map de HRNet y reescala a 128x128. La heuristica NHWC/NCHW es fragil, aunque en la ejecucion guardada no rompe. |
| U-Net | celdas 29, 31-32 | Si por inspeccion | Si por inspeccion | 4 bajadas + bottleneck, 4 subidas, salida 1 canal. No hay salida guardada de prueba de forma. |
| U-Net++ | celdas 30-32 | Si por inspeccion | Si por inspeccion | Implementacion nested completa hasta `x04`, salida 1 canal. La evaluacion posterior tiene un error de funcion. |
| TransUNet | No aparece | No aplica | No aplica | No hay clase, importacion ni entrenamiento de TransUNet en el notebook. |

## Hallazgos por gravedad

### Alta: el checkpoint no guarda necesariamente el mejor modelo

- Celdas afectadas: 11, 14, 21, 26, 31, 32.
- Problema: `fit_model_resumable` actualiza `best_rmse` cuando mejora la validacion, pero siempre guarda el estado actual al final de cada epoca en el mismo `ckpt_path`. Eso significa que el archivo contiene el modelo de la ultima epoca ejecutada, no necesariamente el modelo con mejor RMSE de validacion.
- Impacto: `best_val_RMSE` puede referirse a una epoca anterior, mientras que `test_MAE`, `test_RMSE` y `test_R2` se calculan con pesos de otra epoca. Esto puede explicar diferencias grandes y resultados aparentemente incoherentes.
- Correccion propuesta: guardar dos checkpoints: `last_*.pth` para reanudar y `best_*.pth` solo cuando `val_RMSE` mejora. Evaluar test exclusivamente con `best_*.pth`.

### Alta: nombres de checkpoints incompatibles para U-Net y U-Net++

- Celdas afectadas: 31, 32.
- Problema: en la celda 31 se entrenan con `unet_50ep.pth` y `unetpp_50ep.pth`, pero en la celda 32 se cargan `best_unet.pth` y `best_unetpp.pth`.
- Impacto: la celda 32 puede fallar, cargar checkpoints antiguos de otra ejecucion o evaluar pesos que no corresponden al entrenamiento actual.
- Correccion propuesta: unificar convencion de nombres, por ejemplo `last_unet.pth`, `best_unet.pth`, `last_unetpp.pth`, `best_unetpp.pth`, y hacer que entrenamiento y evaluacion usen las mismas rutas.

### Alta: U-Net++ se evalua con una funcion no definida

- Celda afectada: 32.
- Problema: `test_metrics_unetpp = evaluate(model_unetpp, test_loader)` llama a `evaluate`, pero la funcion definida en el notebook actual es `evaluate_global`.
- Impacto: la evaluacion de U-Net++ puede fallar o, si existe una funcion `evaluate` residual en la sesion de Colab, puede usar una metrica distinta a la del resto de modelos.
- Correccion propuesta: cambiar a `evaluate_global(model_unetpp, test_loader)` y reiniciar el runtime antes de ejecutar de principio a fin.

### Alta: resultados antiguos mezclados con resultados nuevos

- Celdas afectadas: 16, 22, 27, 28, 32.
- Problema: las salidas guardadas muestran columnas `test_loss` que ya no salen de `evaluate_global`, y los resultados se van acumulando en una lista `results` dependiente del orden de ejecucion. Ademas, U-Net/U-Net++ se agregan mas tarde sobre una lista previa.
- Impacto: `results.csv` puede mezclar metricas calculadas con distintas funciones, distintos checkpoints o distintas versiones del codigo.
- Correccion propuesta: crear una unica celda de evaluacion limpia que reconstruya `results = []`, cargue cada `best_*.pth`, use siempre `evaluate_global`, y escriba un CSV nuevo con una columna `run_id` o fecha.

### Alta: preprocesado y escala de metricas no documentados suficientemente

- Celdas afectadas: 6-8, 10.
- Problema: el notebook carga `xtrain_nyc-002.npy`, `ytrain_nyc.npy`, `xtest_nyc.npy`, `ytest_nyc.npy`. Convierte a `float32` y permuta `NHWC -> NCHW`, pero no normaliza de forma explicita. `x_train` tiene min/max aproximado `-3.79 / 16.885`, lo que sugiere que algunos canales podrian estar ya transformados, pero no se comprueba por canal. `y_train` va de `0` a `405.84`, aparentemente en metros o unidades DSM originales.
- Impacto: si el paper entreno con normalizacion por canal y/o DSM escalado, el entrenamiento actual no es directamente comparable. Si `y` esta normalizado en otra version de datos, MAE/RMSE requeririan desnormalizacion antes de compararse con el paper.
- Correccion propuesta: imprimir y guardar media, desviacion, minimo y maximo por canal para train/val/test; documentar si `xtrain_nyc-002.npy` ya esta normalizado; definir explicitamente `y_scale`/`y_offset` si se escala el DSM; calcular metricas finales siempre en metros.

### Media: U-Net y U-Net++ son razonables, pero no necesariamente equivalentes al paper

- Celdas afectadas: 29, 30.
- U-Net actual: 5 niveles efectivos `64,128,256,512,1024`, 4 operaciones de pooling, 4 subidas, dos convoluciones 3x3 por bloque, BatchNorm + ReLU, skip connections por concatenacion, upsampling bilineal por defecto, salida `Conv2d(64,1,1)` sin activacion final.
- U-Net++ actual: niveles `64,128,256,512,1024`, bloques nested `x00` a `x04`, dos convoluciones 3x3 por bloque, BatchNorm + ReLU, upsampling bilineal, salida 1x1 sin activacion final, sin deep supervision.
- Posibles diferencias frente al paper: el articulo accesible publicamente confirma que compara U-Net y U-Net++, pero no permite verificar aqui todos los hiperparametros internos. Si el paper uso transposed convolutions, ausencia/presencia distinta de BatchNorm, deep supervision en U-Net++, otra base de filtros, otra funcion de perdida o otra normalizacion, la comparacion no es estrictamente equivalente.
- Correccion propuesta: crear una tabla `paper_config_vs_notebook` con filtros, niveles, BN, dropout, upsampling, loss, optimizer, epochs, batch size y preprocessing exactos extraidos de la seccion de metodos del paper.

### Media: funcion de perdida distinta a la metrica principal

- Celdas afectadas: 10, 11.
- Problema: se entrena con `SmoothL1Loss(beta=1.0)` sobre la escala del DSM. El paper reporta MAE/RMSE/R2; no queda documentado si entreno con MSE, MAE, Huber u otra perdida.
- Impacto: Huber puede ser robusta a outliers, pero sobre DSM en metros puede penalizar distinto edificios altos y errores grandes. Esto afecta especialmente RMSE.
- Correccion propuesta: ejecutar una comparacion controlada para U-Net/U-Net++ con la perdida del paper si esta especificada, y si no, probar `MSELoss`, `L1Loss` y `SmoothL1Loss` manteniendo todo lo demas fijo.

### Media: split de validacion aleatorio por patch

- Celdas afectadas: 8.
- Problema: `random_split` separa 10% de `x_train_t` para validacion de forma aleatoria. No se comprueba si los patches proceden de zonas espaciales separadas.
- Impacto: si hay patches solapados o muy cercanos, la validacion puede ser optimista y no representar el test. Si el paper uso split espacial o una particion distinta, los resultados no son comparables.
- Correccion propuesta: documentar origen geografico de cada patch y, si es posible, usar split espacial reproducible. Mantener test separado como ahora, pero registrar exactamente la particion.

### Media: HRNet usa solo el ultimo feature map

- Celdas afectadas: 24-26.
- Problema: `HRNetRegressor` usa `feats[-1]` y una cabeza ligera, perdiendo parte de la naturaleza multi-resolucion de HRNet.
- Impacto: no es un HRNet decoder completo para prediccion densa; puede rendir peor y no ser comparable con arquitecturas encoder-decoder.
- Correccion propuesta: fusionar features multi-escala antes de la cabeza, reescalando todos a 128x128 o a una resolucion comun.

### Media: Swin-UNet es un decoder simple sobre backbone Swin, no necesariamente el Swin-UNet canonico

- Celdas afectadas: 12-15.
- Problema: la clase usa `swin_tiny_patch4_window7_224` como encoder de `timm` y tres bloques decoder con interpolacion bilineal. No implementa necesariamente el decoder simetrico y patch-expanding tipico de Swin-UNet.
- Impacto: el nombre `Swin-UNet` puede inducir a comparar contra una arquitectura distinta.
- Correccion propuesta: renombrar en resultados a `Swin encoder + UNet decoder` o implementar una version Swin-UNet canonica si ese es el objetivo experimental.

### Baja: la heuristica NHWC/NCHW de HRNet es fragil

- Celda afectada: 24.
- Problema: la deteccion `if f.shape[1] != 128 and f.shape[-1] > 8` seguida de `if f.shape[-1] > 8 and f.shape[1] < 8` no detecta todos los casos NHWC.
- Impacto: ahora la salida guardada indica que funciona, probablemente porque `timm` devuelve NCHW para ese HRNet. Pero puede romper con otros backbones.
- Correccion propuesta: usar informacion de `feature_info.channels()` para validar si el eje de canales coincide con `last_ch`; si no coincide y el ultimo eje si coincide, permutar.

### Baja: no hay activacion final, lo cual parece correcto para regresion DSM

- Celdas afectadas: 12, 19, 24, 29, 30.
- Observacion: los modelos devuelven una salida lineal sin `sigmoid` ni `relu`.
- Impacto: para regresion de alturas es razonable. El riesgo es que puedan aparecer alturas negativas.
- Correccion propuesta: mantener salida lineal salvo que el paper indique una activacion o escalado especifico. Si aparecen predicciones negativas, analizar si conviene `clamp` solo para metricas/visualizacion, no necesariamente durante entrenamiento.

## Revision del preprocesado

- Orden de canales: correcto a nivel de forma. Los datos cargan como `(N,128,128,4)` y se convierten con `permute(0,3,1,2)` a `(N,4,128,128)`.
- DSM objetivo: correcto a nivel de forma. `y` carga como `(N,128,128,1)` y pasa a `(N,1,128,128)`.
- Normalizacion de entrada: no hay normalizacion explicita en el notebook. Se necesita confirmar si `xtrain_nyc-002.npy` ya contiene bandas normalizadas o transformadas.
- Normalizacion de objetivo: no hay escalado explicito. Las metricas se calculan directamente sobre `y`. Esto es correcto solo si `y` esta en la misma escala que el paper.
- Desnormalizacion: no existe. Si en algun momento se introduce normalizacion de `y`, `evaluate_global` debe desnormalizar `pred` y `target` antes de MAE/RMSE/R2.
- Split: train/validation es aleatorio con seed 42; test viene de arrays separados. Falta documentacion espacial.

## Revision de metricas

- MAE y RMSE se calculan por pixel acumulando suma absoluta, suma cuadratica y `n = y.numel()`. Esto esta bien para metricas globales pixel-wise.
- R2 se calcula globalmente sobre todos los pixeles concatenados. Esto es coherente, aunque puede diferir de un R2 promediado por imagen.
- `evaluate_global` usa `model.eval()` y `@torch.no_grad()`, correcto.
- Hay interpolacion silenciosa si la salida no coincide en resolucion. Esto ayuda a no romper, pero puede ocultar modelos que devuelven resoluciones incorrectas. Para auditoria conviene registrar un warning por modelo.
- Riesgo fuerte: en U-Net++ se llama a `evaluate`, no a `evaluate_global`; eso rompe la consistencia.

## Revision de entrenamiento

- Funcion de perdida: `SmoothL1Loss(beta=1.0)`.
- Optimizador: `AdamW`.
- Learning rate: `1e-4` por defecto.
- Weight decay: `1e-4`.
- Scheduler: `CosineAnnealingLR(T_max=epochs)`, baja hasta 0 al final.
- Batch size: 16.
- Epocas: 50.
- `model.train()`: usado correctamente en entrenamiento.
- `model.eval()` y `torch.no_grad()`: usados correctamente en validacion/test mediante `evaluate_global`.
- Checkpoints: problema alto. No se guarda el mejor estado de pesos, solo el ultimo estado con el mejor RMSE como escalar.

## Inconsistencias del notebook

1. `evaluate` usado antes/no definido en celda 32 para U-Net++.
2. `unet_50ep.pth` y `unetpp_50ep.pth` no coinciden con `best_unet.pth` y `best_unetpp.pth`.
3. Resultados con `test_loss` aparecen en salidas guardadas, pero `evaluate_global` no devuelve `loss`.
4. `results` depende del estado de sesion y del orden de ejecucion.
5. Varias clases se redefinen (`DoubleConv` aparece para Attention U-Net y despues para U-Net), lo que no rompe si se ejecuta en orden, pero complica la trazabilidad.
6. El notebook conserva salidas antiguas de Colab, por lo que no es fiable como registro reproducible sin una ejecucion limpia de principio a fin.

## Posibles causas de la diferencia con el paper

1. Checkpoints evaluados no corresponden al mejor modelo de validacion. Gravedad alta.
2. U-Net/U-Net++ cargan rutas incompatibles o antiguas. Gravedad alta.
3. U-Net++ puede estar evaluandose con una funcion residual distinta. Gravedad alta.
4. Preprocesado no documentado/no equivalente al paper. Gravedad alta.
5. Perdida, scheduler, batch size, epocas o normalizacion distintos al paper. Gravedad media-alta.
6. Validacion aleatoria por patch no necesariamente equivalente al split del paper. Gravedad media.
7. Swin-UNet y HRNet implementados como variantes simplificadas, no necesariamente arquitecturas canonicas. Gravedad media.

## Propuesta de siguiente paso seguro

Antes de refactorizar modelos, conviene hacer una limpieza minima y reproducible:

1. Crear una celda `sanity_check_models` que instancie cada modelo y verifique explicitamente entrada `(2,4,128,128)` y salida `(2,1,128,128)`.
2. Corregir solo nombres de checkpoint y `evaluate_global` para U-Net++.
3. Separar `last_*.pth` y `best_*.pth`.
4. Crear una unica funcion `evaluate_all_best_checkpoints` que regenere `results.csv` desde cero.
5. Añadir una celda de estadisticas por canal y escala de `y`.
6. Ejecutar U-Net y U-Net++ desde runtime limpio como baseline antes de juzgar Attention U-Net, Swin-UNet o HRNet.

Conclusion: con el notebook actual no se puede concluir que las nuevas arquitecturas sean peores o mejores que U-Net/U-Net++ del paper. Primero hay que corregir reproducibilidad, checkpoints, evaluacion y escala de datos. Solo despues la comparacion arquitectonica sera interpretable.
