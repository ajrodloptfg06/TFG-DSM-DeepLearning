# Changelog

## 2026-08-20 — orquestador secuencial del entrenamiento final

Rama: final-training-orchestrator

### Añadido

- MODEL_REGISTRY define el orden oficial, constructor, checkpoints last/best y uso de backbone preentrenado de los cinco modelos.
- La validación del registro impide nombres duplicados y rutas de checkpoint compartidas.
- STRICT_SHAPE_CHECK_PASSED solo se activa después de que los cinco modelos superen el sanity check nativo.
- run_final_training_pipeline entrena un único modelo cada vez, registra best val_RMSE y libera CPU/GPU antes del siguiente.
- RUN_FINAL_TRAINING y RUN_FINAL_EVALUATION son False por defecto y requieren activación explícita.
- evaluate_best_checkpoints comprueba todos los best antes de iniciar la única evaluación oficial sobre test.
- Las antiguas celdas de visualización sobre test quedan desactivadas para evitar consultas fuera de la evaluación oficial.

### Cambiado

- Se sustituyen las celdas de entrenamiento automático/manual por el flujo secuencial protegido.
- El resumen previo muestra las banderas, rutas, hiperparámetros y los cinco modelos registrados.
- FINAL_EXPERIMENT_PROTOCOL.md documenta configuración, datos, sanity check, entrenamiento opt-in, evaluación opt-in y results.csv oficial.

### No cambiado

- Arquitecturas, métricas, loss, optimizador, scheduler y número de épocas.
- El notebook FAST, datasets, checkpoints y resultados existentes.

## 2026-08-04 — auditoría de escala segura en memoria

Rama: memory-safe-data-scale-audit

### Cambiado

- La auditoría de train, validation y test procesa muestras por chunks sin construir subconjuntos completos mediante advanced indexing.
- Count, finite_count, nonfinite_count, zero_fraction, min, max, mean y std se acumulan sobre todos los valores con memoria acotada.
- Los percentiles usan todos los valores finitos hasta MAX_PERCENTILE_SAMPLE y, por encima del límite, un muestreo reproducible; percentiles_method documenta el método por fila.
- RUN_DATA_SCALE_AUDIT es False por defecto y la celda explica cómo activar el análisis manualmente.
- El protocolo final aclara que la auditoría completa es opcional y no bloquea el sanity check.

### No cambiado

- Arquitecturas, métricas, entrenamiento, loss, optimizador, scheduler y épocas.
- El notebook FAST, checkpoints y resultados existentes.

## 2026-08-03 — sanity check independiente

Rama: make-sanity-check-independent

### Cambiado

- El sanity check define localmente las factorías de los cinco modelos.
- El tensor dummy se crea dentro de la función con forma exacta (2,4,128,128).
- La salida se valida de forma nativa contra (2,1,128,128), sin interpolación.
- No se usan modelos entrenados, histories, rutas de checkpoint ni outputs de fit.
- Cada modelo temporal se mueve a CPU, se elimina y libera caché CUDA antes del siguiente.

### No cambiado

- Las clases de los cinco modelos y sus forwards.
- Métricas, loss, optimizador, scheduler y épocas.
- Las celdas de entrenamiento, que siguen creando sus propias instancias.
- El notebook FAST, checkpoints y resultados existentes.

## 2026-08-03 — seguridad del run FINAL

Rama: final-run-safety-fixes

### Añadido

- ALLOW_RESUME, controlado por TFG_ALLOW_RESUME y desactivado por defecto.
- Bloqueo de RUN_ID final cuando ya contiene artefactos y no se autorizó resume.
- Validación de compatibilidad del manifiesto: modo, seed, épocas, batch size, learning rate, weight decay y rutas de datos.
- Metadatos de compatibilidad en checkpoints nuevos, incluida clase de modelo y ruta.
- RUN_INTERMEDIATE_TEST_EVALS, desactivado por defecto.
- Resumen completo de configuración impreso antes del entrenamiento.

### Cambiado

- Las cinco definiciones de modelo preceden a cualquier entrenamiento.
- El sanity check exige la salida nativa exacta (2,1,128,128) sin interpolación.
- Entrenamiento, evaluación y visualización liberan modelos secuencialmente de GPU.
- evaluate_best_checkpoints recibe factorías y evalúa un único modelo cada vez.
- Las consultas intermedias a test solo pueden activarse de forma explícita.
- FINAL_EXPERIMENT_PROTOCOL.md refleja las medidas aplicadas y sus limitaciones.

### Compatibilidad

- No se eliminan checkpoints ni resultados existentes.
- Los checkpoints antiguos sin metadatos compatibility se conservan, pero se rechazan para resume automático.
- No se modifican arquitecturas, métricas, loss, AdamW, CosineAnnealingLR ni las 50 épocas de FINAL.

## Unreleased — aislamiento y trazabilidad de ejecuciones

Rama: `fix-run-isolation-and-traceability`

### Alcance cerrado de modelos

- Baselines: U-Net y U-Net++.
- Alternativas: Attention U-Net residual, Swin híbrido con decoder CNN y HRNet multiescala.
- TransUNet queda explícitamente descartado y no se implementará.
- No se añadirán variantes canónicas adicionales en esta fase; el trabajo se centra en limpiar y trazar la comparación de estos cinco modelos.
### Añadido

- Configuración central de ejecución en ambos notebooks:
  - `RUN_MODE` (`fast` o `final`);
  - `RUN_ID` único, con override opcional mediante `TFG_RUN_ID`;
  - `EXPERIMENT_NAME`;
  - `SEED`;
  - `MODEL_NAME`.
- Validación estricta del modo esperado por notebook. El notebook FAST rechaza cualquier intento de ejecutarse como FINAL y viceversa.
- Validación de `RUN_ID` para impedir rutas relativas o identificadores inseguros.
- Directorios aislados por modo y ejecución:
  - `checkpoints/<run_mode>/<run_id>/`;
  - `results/<run_mode>/<run_id>/`.
- Guardas explícitas para impedir que FAST escriba bajo `checkpoints/final` o `results/final`.
- Helper `checkpoint_paths(model_slug)` para generar nombres `last_<model>.pth` y `best_<model>.pth` dentro del run activo.
- `run_manifest.json` por ejecución, escrito de forma atómica en la carpeta de resultados. Incluye:
  - identidad y fecha del run;
  - commit Git cuando está disponible;
  - seed e hiperparámetros principales;
  - modelo/checkpoint activo y registro de checkpoints por modelo;
  - rutas de datos esperadas;
  - versiones disponibles de Python, PyTorch, timm, NumPy y pandas;
  - CSV y fecha de evaluación cuando finaliza la evaluación oficial.
- Columnas de trazabilidad en `results.csv`: `run_id`, `run_mode`, `seed` y `epochs`, además de modelo, checkpoint y métricas.

### Cambiado

- Los DataLoader usan ahora `CFG["batch_size"]`, `CFG["num_workers"]` y `SEED`.
- Las rutas de datos se centralizan en `DATA_PATHS` sin cambiar los nombres ni el contenido esperado de los arrays.
- La evaluación oficial exige todos los mejores checkpoints y regenera `results.csv` desde cero con un orden de columnas estable.
- U-Net y U-Net++ siguen cargando sus mejores pesos para la visualización posterior, pero esa celda ya no acumula ni guarda resultados.

### Desactivado

- Flujo mutable `results = []` + `results.append(...)`.
- Escrituras intermedias de `results.csv` que podían dejar archivos parciales o duplicados.

### No cambiado

- Arquitecturas y forwards de U-Net, U-Net++, Attention U-Net, Swin y HRNet.
- Fórmulas de MAE, RMSE y R².
- Loss, optimizador y scheduler.
- Checkpoints, datasets y resultados existentes en Drive: no se borran ni se migran.
- Normalización y equivalencia de escala con el paper, que permanecen como trabajo posterior de la auditoría.

### Compatibilidad

Los artefactos antiguos no se borran ni sobrescriben. Tras final-run-safety-fixes, reanudar exige TFG_ALLOW_RESUME=true y metadatos compatibility; los checkpoints antiguos sin ellos se conservan pero se rechazan de forma segura.
