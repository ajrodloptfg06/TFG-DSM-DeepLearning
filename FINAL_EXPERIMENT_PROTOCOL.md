# Protocolo del experimento final

Fecha de revisión: 2026-08-25

Rama: **final-experiment-protocol**

Notebook principal revisado: **TFG_DsmV6.ipynb**

## 1. Estado del notebook principal

### 1.1 Configuración FINAL confirmada

El notebook principal configura de forma explícita:

- **RUN_MODE = final**.
- **FAST_DEV_RUN = False**, porque se calcula como RUN_MODE == fast.
- **50 épocas** en lugar de las 2 épocas del notebook FAST.
- Backbones preentrenados para Swin y HRNet mediante pretrained_backbones=True.
- Checkpoints en **PROJECT_ROOT/checkpoints/final/RUN_ID/**.
- Resultados en **PROJECT_ROOT/results/final/RUN_ID/**.
- Un RUN_ID único por defecto con fecha UTC, hora y sufijo UUID.
- Un manifiesto en **results/final/RUN_ID/run_manifest.json**.

La variable de entorno TFG_RUN_MODE se comprueba contra el modo fijo del notebook. Si contiene fast, el notebook principal falla en vez de escribir silenciosamente como FINAL.

### 1.2 Aislamiento de artefactos

Los cinco modelos obtienen sus rutas mediante checkpoint_paths y comparten exclusivamente el directorio de su ejecución:

- last_swin_unet.pth / best_swin_unet.pth;
- last_attention_unet_residual.pth / best_attention_unet_residual.pth;
- last_hrnet_w18_multiscale.pth / best_hrnet_w18_multiscale.pth;
- last_unet.pth / best_unet.pth;
- last_unetpp.pth / best_unetpp.pth.

La evaluación oficial carga esos cinco checkpoints best desde el mismo CKPT_DIR y solo escribe results.csv después de haber evaluado correctamente todos los modelos. No usa results=[] ni append entre celdas.

### 1.3 Veredicto de preparación tras los cambios de seguridad

**El notebook principal queda preparado estructuralmente para una ejecución FINAL limpia de arriba abajo**, sujeto a las comprobaciones externas de datos, escala, dependencias y recursos descritas en este protocolo.

En la rama final-run-safety-fixes se han aplicado estas medidas:

1. Las cinco arquitecturas se definen antes de cualquier entrenamiento.
2. El sanity check conjunto se ejecuta antes del primer modelo, exige la salida nativa exacta (2,1,128,128) y no interpola.
3. Las evaluaciones intermedias sobre test están protegidas por RUN_INTERMEDIATE_TEST_EVALS=False.
4. Cada modelo se libera de GPU antes de construir el siguiente.
5. La evaluación oficial usa factorías y mantiene un único modelo en GPU.
6. ALLOW_RESUME es False por defecto y un RUN_ID final con artefactos existentes falla antes de escribir.
7. La reanudación explícita valida manifiesto y metadatos del checkpoint.
8. La configuración efectiva se imprime antes de cargar datos y entrenar.

Permanecen riesgos que no pertenecen a este bloque: unidades y nodata del DSM, equivalencia del split con el paper, versiones no fijadas de timm/PyTorch, una sola seed y ausencia de fingerprints del contenido de los arrays.

## 2. A. Objetivo del experimento final

Ejecutar una comparación controlada, reproducible y trazable entre las dos baselines del paper y los tres modelos alternativos seleccionados para el TFG.

La pregunta experimental es:

> Bajo el mismo dataset, split, preprocesado, presupuesto de épocas y evaluación, ¿cómo se comparan Attention U-Net residual, el híbrido Swin y HRNet multiescala frente a U-Net y U-Net++?

El objetivo no es demostrar por anticipado que una arquitectura alternativa sea mejor. La conclusión debe depender exclusivamente de los resultados regenerados desde los mejores checkpoints de la misma ejecución FINAL.

## 3. B. Modelos incluidos

| Grupo | Etiqueta recomendada | Clase | Checkpoint best |
|---|---|---|---|
| Baseline | U-Net | UNet | best_unet.pth |
| Baseline fuerte | U-Net++ | UNetPlusPlus | best_unetpp.pth |
| Alternativo | Attention U-Net residual | AttentionUNet | best_attention_unet_residual.pth |
| Alternativo | Swin-Tiny encoder + decoder CNN | SwinUNet | best_swin_unet.pth |
| Alternativo | HRNet-W18-Small multiescala | HRNetRegressor | best_hrnet_w18_multiscale.pth |

No se incluye TransUNet y no debe aparecer en el entrenamiento, los checkpoints, results.csv ni las conclusiones.

## 4. C. Configuración común

Configuración efectiva observada en el notebook principal:

| Elemento | Configuración |
|---|---|
| Entrada | Tensor float32 NCHW de forma (B,4,128,128) |
| Objetivo | DSM float32 NCHW de forma (B,1,128,128) |
| Batch size | 16 |
| Épocas | 50 |
| Seed registrado | 42 |
| Loss | SmoothL1Loss con beta=1.0 |
| Optimizador | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| T_max | Número total de épocas: 50 |
| Paso del scheduler | Una vez después de cada época |
| Métricas de selección | RMSE de validación |
| Métricas finales | test_MAE, test_RMSE y test_R2 |
| Loss final informado | test_loss con SmoothL1Loss |
| Split de validación | 90 % train / 10 % validation mediante random_split |
| Test | Arrays x_test e y_test separados |
| Preprocesado visible | Conversión a float32 y permutación NHWC → NCHW |
| Normalización visible | Ninguna normalización o desnormalización explícita |
| Backbones preentrenados en FINAL | Sí para Swin y HRNet |

MAE, RMSE y R² se agregan globalmente sobre todos los píxeles del loader. El best checkpoint se selecciona por el menor RMSE de validación.

## 5. D. Condiciones que deben mantenerse idénticas

Para que la comparación sea defendible, los cinco modelos deben compartir:

- los mismos cuatro canales y su mismo orden;
- exactamente los mismos archivos de datos;
- el mismo split train/validation, incluidos sus índices;
- el mismo test set;
- la misma conversión de tipos y layout;
- cualquier normalización, clipping, máscara o tratamiento nodata;
- el mismo batch size;
- el mismo número máximo de épocas;
- la misma loss;
- el mismo optimizador, learning rate y weight decay;
- el mismo scheduler y frecuencia de step;
- la misma seed declarada y una política reproducible de RNG;
- la misma función evaluate_global;
- la selección del checkpoint por val_RMSE;
- la evaluación final exclusivamente desde best;
- las mismas unidades para predicción y target.

La igualdad del protocolo no exige que las arquitecturas tengan el mismo número de parámetros ni que todos los modelos partan del mismo tipo de inicialización. La utilización de preentrenamiento en Swin y HRNet debe declararse como parte del diseño experimental.

## 6. E. Procedimiento recomendado en Colab

### 6.1 Preparación

1. Fusionar esta documentación cuando haya sido revisada, sin modificar el notebook durante el run.
2. Abrir **TFG_DsmV6.ipynb** desde la rama main y anotar el SHA exacto de main.
3. Iniciar un runtime limpio de Colab; no reutilizar una sesión con clases, variables o modelos antiguos.
4. Seleccionar GPU y comprobar nvidia-smi.
5. Montar Google Drive.
6. Confirmar PROJECT_ROOT y DATA_ROOT antes de crear artefactos.
7. Confirmar que RUN_MODE imprime final y que FAST_DEV_RUN es False.
8. Usar un RUN_ID nuevo. No reutilizar el identificador de un run anterior para una comparación limpia.
9. Si se proporciona TFG_RUN_ID, TFG_EXPERIMENT_NAME o GIT_COMMIT, hacerlo antes de ejecutar la celda de configuración.
10. Confirmar que CKPT_DIR contiene checkpoints/final/RUN_ID y RES_DIR contiene results/final/RUN_ID.

### 6.2 Datos

11. Comprobar que existen los cuatro archivos esperados: x_train, y_train, x_test e y_test.
12. Ejecutar la carga y verificar shapes, dtype, min y max.
13. Decidir si se necesita regenerar la auditoría completa de escala. Es opcional para arrancar el run final: mantener RUN_DATA_SCALE_AUDIT=False para omitirla o activarla manualmente; la implementación por chunks evita copias completas y marca los percentiles aproximados.
14. Confirmar que no hay NaN o infinitos.
15. Confirmar cómo se representa nodata.
16. Confirmar por escrito las unidades y el datum del DSM antes de interpretar MAE/RMSE como metros.
17. Conservar evidencia de los índices del split o un fingerprint que permita reconstruirlo.

### 6.3 Sanity checks

18. Ejecutar un sanity check nativo de cada modelo con entrada (2,4,128,128).
19. Verificar la salida antes de cualquier interpolación correctiva.
20. Exigir exactamente (2,1,128,128) en los cinco modelos.
21. Registrar canales y layouts reales de timm para Swin y HRNet.
22. No iniciar entrenamiento si algún modelo falla, si las dependencias cambian el layout o si la GPU no tiene memoria suficiente.

El notebook actual ya coloca las cinco definiciones y el shape check estricto antes del primer entrenamiento. El check libera cada modelo antes de construir el siguiente.

La auditoría de escala no es una dependencia del sanity check. Con RUN_DATA_SCALE_AUDIT=False, la celda solo informa cómo activarla y la ejecución puede continuar directamente a las definiciones y al shape check estricto. Si aún faltan evidencias de unidades, nodata o escala, deben obtenerse y documentarse antes de interpretar los resultados, pero su cálculo no debe bloquear una comprobación de shapes.

### 6.4 Entrenamiento secuencial oficial

23. Mantener RUN_FINAL_TRAINING=False mientras se ejecutan configuración, datos, definiciones y sanity check.
24. Confirmar que STRICT_SHAPE_CHECK_PASSED es True y revisar el resumen impreso del MODEL_REGISTRY.
25. Cambiar explícitamente RUN_FINAL_TRAINING=True en la celda resumen/control situada después del sanity check; no reejecutar la configuración central ni regenerar RUN_ID.
26. Ejecutar run_final_training_pipeline una sola vez. El registro fija el orden U-Net, U-Net++, Attention U-Net residual, Swin híbrido y HRNet multiescala.
27. Verificar para cada modelo que se actualizan last y best en CKPT_DIR y que el mejor val_RMSE queda en el resumen y el manifiesto.
28. No copiar checkpoints de otra carpeta ni renombrar artefactos manualmente.
29. Si Colab se interrumpe, reanudar solo con el mismo RUN_ID y después de comprobar que manifiesto, configuración, datos y commit siguen siendo idénticos.
30. No consultar test para decidir hiperparámetros, épocas, arquitectura o selección de checkpoint. El orquestador solo recibe train_loader y val_loader.
31. Confirmar que cada modelo se libera antes de instanciar el siguiente.

### 6.5 Evaluación oficial

32. Mantener RUN_FINAL_EVALUATION=False durante todo el entrenamiento.
33. Comprobar que existen los cinco best checkpoints.
34. Cambiar explícitamente RUN_FINAL_EVALUATION=True en la celda resumen/control, ejecutarla para registrar el estado y después ejecutar la celda oficial de evaluación una sola vez.
35. evaluate_best_checkpoints valida primero la presencia de los cinco best, construye un único modelo cada vez y falla antes de consultar test si falta alguno.
36. Confirmar que cada fila usa el checkpoint best del mismo RUN_ID.
37. Guardar results.csv en results/final/RUN_ID.
38. Confirmar que run_manifest.json se actualiza con los checkpoints y la ruta del CSV.
39. Tratar results.csv regenerado por esa función como única tabla oficial.
40. Mantener desactivadas las visualizaciones sobre test dentro del flujo oficial; realizarlas, si se necesitan, como análisis posterior a partir de un protocolo separado y sin alterar results.csv.

## 7. F. Archivos que debe generar cada run

Directorio esperado:

    PROJECT_ROOT/
    ├── checkpoints/
    │   └── final/
    │       └── RUN_ID/
    │           ├── last_swin_unet.pth
    │           ├── best_swin_unet.pth
    │           ├── last_attention_unet_residual.pth
    │           ├── best_attention_unet_residual.pth
    │           ├── last_hrnet_w18_multiscale.pth
    │           ├── best_hrnet_w18_multiscale.pth
    │           ├── last_unet.pth
    │           ├── best_unet.pth
    │           ├── last_unetpp.pth
    │           └── best_unetpp.pth
    └── results/
        └── final/
            └── RUN_ID/
                ├── results.csv
                └── run_manifest.json

Los archivos last sirven para reanudar. Los archivos best sirven para la evaluación y comparación oficial. No deben intercambiarse sus funciones.

results.csv debe contener exactamente cinco filas y, como mínimo:

- run_id;
- run_mode;
- model;
- checkpoint;
- best_val_RMSE;
- test_loss;
- test_MAE;
- test_RMSE;
- test_R2;
- seed;
- epochs.

run_manifest.json debe conservar la identidad del run, hiperparámetros, rutas de datos esperadas, versiones disponibles, checkpoints usados y ruta del CSV.

## 8. G. Criterio de comparación

1. Verificar primero que las cinco filas corresponden al mismo run_id y run_mode=final.
2. Ordenar la tabla por **test_RMSE ascendente**.
3. Usar test_RMSE como criterio principal porque penaliza especialmente errores grandes.
4. Comparar también **test_MAE ascendente** y **test_R2 descendente**.
5. Considerar U-Net++ como baseline fuerte y comparar cada alternativa tanto con U-Net como con U-Net++.
6. Informar best_val_RMSE para comprobar coherencia entre selección y generalización, sin sustituir las métricas de test.
7. No afirmar mejora si los modelos no comparten test set, split, preprocesado, unidades, seed/protocolo y función de evaluación.
8. No presentar una diferencia pequeña como concluyente con una única seed.
9. No seleccionar el modelo final por una métrica calculada sobre una muestra visual individual.

Una tabla ordenada ayuda a presentar los resultados, pero no debe sobrescribir results.csv. El CSV original del run debe preservarse como evidencia.

## 9. H. Riesgos antes de ejecutar

### Escala y datos

- Las unidades físicas de y siguen pendientes de confirmación documental.
- No hay normalización explícita de x ni y.
- No existe desnormalización antes de MAE/RMSE.
- El tratamiento nodata o máscaras no está demostrado.
- El split aleatorio por patch puede no coincidir con el paper ni impedir fuga espacial.
- El orden y significado físico de los cuatro canales debe quedar documentado.

### Dependencias y modelos

- timm no está fijado a una versión.
- Swin presupone cuatro features NHWC y las convierte incondicionalmente a NCHW.
- HRNet depende del conjunto de features y canales informado por timm.
- Los pesos preentrenados de tres canales se adaptan a cuatro mediante timm; debe registrarse su comportamiento.
- Swin es un híbrido encoder Transformer + decoder CNN, no un Swin-UNet canónico.
- Attention U-Net usa gates residuales.
- HRNet usa una cabeza multiescala propia.

### Protocolo estadístico

- Solo se ha definido una seed.
- El RNG no se reinicia por modelo en el flujo actual.
- No hay intervalos de confianza ni variabilidad entre runs.
- La comparación con el paper puede no ser directa por datos, split, preprocessing o unidades.

### Ejecución

- Reutilizar RUN_ID solo está permitido con TFG_ALLOW_RESUME=true y compatibilidad validada; para un experimento limpio debe usarse un identificador nuevo.
- Una reanudación autorizada puede conservar un results.csv previo hasta repetir la evaluación oficial; debe comprobarse su fecha al finalizar.
- RUN_INTERMEDIATE_TEST_EVALS permite depuración explícita; debe permanecer false para no consultar test antes de la evaluación oficial.
- Los modelos se liberan secuencialmente, aunque Swin y HRNet todavía pueden superar la memoria disponible por sí solos.
- El sanity check conjunto es previo y estricto; cualquier incumplimiento nativo detiene el run.
- Abrir el notebook desde GitHub no garantiza que git rev-parse funcione en Colab; el commit puede quedar sin registrar.
- Los notebooks versionados no contienen outputs de una ejecución final completa.

## 10. I. Checklist antes del experimento final

- [ ] Estoy usando TFG_DsmV6.ipynb desde main y he anotado su SHA.
- [ ] El runtime de Colab está limpio.
- [ ] La GPU está disponible y tiene memoria suficiente.
- [ ] Drive está montado en la cuenta y ruta correctas.
- [ ] RUN_MODE es final.
- [ ] FAST_DEV_RUN es False.
- [ ] CFG epochs es 50.
- [ ] CFG batch_size es 16.
- [ ] El RUN_ID es nuevo y no existe en checkpoints/final ni results/final.
- [ ] EXPERIMENT_NAME identifica claramente la comparación.
- [ ] GIT_COMMIT quedará registrado o se anotará manualmente.
- [ ] Los cuatro archivos de datos existen.
- [ ] Shapes de x e y son los esperados.
- [ ] No hay NaN ni infinitos.
- [ ] Se ha confirmado nodata/máscaras.
- [ ] Se han confirmado unidades y datum del DSM.
- [ ] Se ha guardado evidencia del split train/validation/test.
- [ ] Las versiones de Python, PyTorch, timm, NumPy y pandas están registradas.
- [ ] Los cinco modelos pasan el shape check nativo.
- [ ] Swin y HRNet informan los layouts y canales esperados.
- [ ] No hay checkpoints copiados manualmente dentro del nuevo RUN_ID.
- [ ] Se acepta explícitamente que este run usa una única seed.

Si no se pueden marcar las comprobaciones de escala, nodata, split o shapes, el entrenamiento puede servir como prueba técnica, pero no debería declararse todavía experimento final comparable.

## 11. J. Checklist después del experimento final

- [ ] Existen cinco checkpoints last.
- [ ] Existen cinco checkpoints best.
- [ ] Cada best contiene model, epoch, best_rmse e history.
- [ ] evaluate_best_checkpoints terminó sin errores.
- [ ] results.csv existe dentro de results/final/RUN_ID.
- [ ] results.csv contiene exactamente cinco modelos sin duplicados.
- [ ] Todas las filas tienen el mismo run_id.
- [ ] Todas las filas tienen run_mode=final.
- [ ] Los nombres de checkpoint coinciden con el modelo.
- [ ] No hay métricas NaN o infinitas.
- [ ] run_manifest.json existe y es JSON válido.
- [ ] El manifiesto contiene las cinco rutas best realmente usadas.
- [ ] El manifiesto contiene versiones, seed, épocas y rutas de datos.
- [ ] El SHA del código está registrado o documentado externamente.
- [ ] Se ha copiado el CSV a una ubicación de respaldo sin editar el original.
- [ ] La tabla de presentación se ha ordenado por test_RMSE.
- [ ] Se han comparado también test_MAE y test_R2.
- [ ] Las conclusiones distinguen evidencia, hipótesis y limitaciones.
- [ ] No se afirma superioridad estadística con una sola seed.
- [ ] No se han mezclado artefactos FAST ni otros RUN_ID.

## 12. Análisis de riesgo de mezcla

| Riesgo | Estado actual | Condición segura |
|---|---|---|
| Checkpoints FAST en FINAL | Aislados por RUN_MODE | Usar el notebook principal sin cambiar rutas |
| Resultados FAST en FINAL | Aislados por RUN_MODE | Verificar RES_DIR antes de entrenar |
| Checkpoints de otro FINAL | Posible si se reutiliza RUN_ID | Crear un RUN_ID nuevo |
| Resultados de otro FINAL | Posible si se reutiliza RUN_ID | Confirmar que results/final/RUN_ID no existe |
| CSV parcial del mismo run | La función escribe tras evaluar los cinco | No reutilizar un directorio que ya tenga CSV |
| Lista acumulativa | Desactivada | Usar solo evaluate_best_checkpoints |
| Outputs versionados antiguos | No hay outputs guardados actualmente | Empezar con runtime limpio |
| Variables residuales de Colab | Posibles si no se reinicia | Reiniciar runtime antes del run |
| Checkpoint last frente a best | Separados | Evaluar exclusivamente best |
| Modelos de commits distintos | No se valida al reanudar | Registrar SHA y no reanudar tras cambiar código |

## 13. Cambios de seguridad aplicados

La rama final-run-safety-fixes aplica los cambios mínimos de seguridad sin modificar arquitecturas, métricas, loss, optimizador, scheduler ni épocas.

### Aplicado

1. **Definiciones antes del entrenamiento.** U-Net, U-Net++, Attention U-Net residual, Swin híbrido y HRNet multiescala quedan disponibles antes del sanity check y de cualquier llamada de entrenamiento.
2. **Shape check nativo y estricto.** Se prueba cada factoría con (2,4,128,128), se exige (2,1,128,128) y no se corrige la salida mediante interpolación.
3. **Independencia del entrenamiento.** El sanity check usa únicamente constructores del registro y su propio tensor dummy; no usa modelos entrenados, histories ni outputs de fit. La marca STRICT_SHAPE_CHECK_PASSED solo se activa tras validar los cinco modelos.
4. **Test intermedio desactivado.** RUN_INTERMEDIATE_TEST_EVALS es False por defecto. evaluate_best_checkpoints sigue siendo la única evaluación oficial.
5. **Memoria secuencial.** Los modelos se mueven a CPU, se eliminan sus referencias, se ejecuta gc.collect() y, cuando hay CUDA, torch.cuda.empty_cache().
6. **Evaluación mediante factorías.** La función oficial construye, evalúa y libera un modelo cada vez.
7. **RUN_ID protegido.** En FINAL, ALLOW_RESUME=False falla si las carpetas del RUN_ID ya contienen artefactos.
8. **Resume explícito y compatible.** TFG_ALLOW_RESUME=true exige coincidencia de run_id, run_mode, seed, épocas, batch size, learning rate, weight decay y rutas esperadas. Cada checkpoint nuevo registra además clase de modelo y ruta.
9. **Configuración visible.** Antes de entrenar se imprimen modo, identificador, política de resume, directorios e hiperparámetros principales.
10. **Visualización fuera del flujo oficial.** Las celdas que cargaban un modelo y una muestra de test quedan desactivadas; evaluate_best_checkpoints es el único consumidor de test durante este protocolo.
11. **Auditoría de escala opcional y acotada.** RUN_DATA_SCALE_AUDIT es False por defecto. Cuando se activa, las estadísticas exactas se acumulan por chunks y los percentiles se limitan mediante MAX_PERCENTILE_SAMPLE; el sanity check no depende del DataFrame resultante.
12. **Registro único.** MODEL_REGISTRY contiene los cinco modelos en orden oficial, constructores, checkpoints last/best y uso efectivo de backbone preentrenado; valida nombres y rutas duplicadas.
13. **Entrenamiento opt-in.** RUN_FINAL_TRAINING es False por defecto. run_final_training_pipeline exige modo FINAL, rutas del RUN_ID y sanity check correcto, y entrena/libera un solo modelo cada vez.
14. **Evaluación opt-in.** RUN_FINAL_EVALUATION es False por defecto. evaluate_best_checkpoints valida los cinco best antes de consultar test y regenera results.csv y run_manifest.json.

### Limitaciones conocidas

- Los checkpoints creados antes de esta protección no contienen el bloque compatibility. Se conservan, pero no se reanudan automáticamente.
- Las rutas de datos se comparan, pero todavía no se verifican hashes del contenido de los arrays.
- La compatibilidad verifica la clase del modelo y que el state_dict pueda cargarse; no calcula un hash de la definición Python.
- TFG_RUN_INTERMEDIATE_TEST_EVALS permite depuración explícita, pero debe permanecer false durante el experimento final.
- Siguen pendientes las unidades de y, nodata, equivalencia espacial del split, versiones fijadas y múltiples seeds.

## 14. Conclusión operativa

El notebook separa FAST y FINAL, protege la creación de runs finales, valida la compatibilidad mínima al reanudar, comprueba shapes nativos antes de entrenar y orquesta secuencialmente un único modelo en GPU. Entrenamiento y evaluación permanecen desactivados por defecto. La tabla oficial continúa regenerándose exclusivamente desde los cinco mejores checkpoints.

Por tanto, el pipeline queda **listo a nivel de orden, aislamiento, memoria y evaluación** para el run FINAL. Antes de iniciarlo todavía deben confirmarse los datos, las unidades del DSM, nodata, la versión de timm/PyTorch, el SHA del código y la capacidad de la GPU.

## 15. Validación repetida y relación con el resultado final

El notebook incorpora un flujo opcional de *repeated holdout* con cinco splits aleatorios reproducibles 80/20 sobre `x_train` / `y_train`. Está desactivado por defecto mediante `RUN_CROSS_VALIDATION=False` y se configura inicialmente solo para Attention U-Net residual, evitando iniciar 25 entrenamientos de forma accidental.

Este análisis responde a una pregunta complementaria: cuánto varían MAE, RMSE y R² de validación al cambiar la partición train/validation. No es un 5-fold clásico disjunto y no consulta el conjunto de test.

La jerarquía de evidencia del TFG queda así:

1. El run FINAL sobre el test reservado sigue siendo la comparación principal entre los cinco modelos.
2. La validación repetida aporta media y desviación estándar para medir estabilidad frente al split.
3. Las métricas de validación repetida no se mezclan, promedian ni sustituyen a las métricas finales de test.

Los checkpoints y resultados se aíslan bajo `checkpoints/crossval/<RUN_ID>/` y `results/crossval/<RUN_ID>/`. El procedimiento completo, las etiquetas válidas de modelos y las condiciones de reanudación se describen en `CROSS_VALIDATION.md`.
