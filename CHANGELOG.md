# Changelog

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

Los artefactos antiguos situados directamente en los directorios anteriores no se borran ni sobrescriben. Para reanudar un run concreto debe reutilizarse su mismo `RUN_ID`, preferentemente definiendo `TFG_RUN_ID` antes de ejecutar la configuración.
