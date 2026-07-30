# Auditoría de datos, escala y comparabilidad con el paper

Fecha: 2026-07-30

Rama: data-scale-and-paper-comparability

## 1. Alcance y evidencia disponible

Esta auditoría revisa estáticamente los dos notebooks versionados y la documentación existente. No se dispone de los arrays de Google Drive en el repositorio, por lo que no se han recalculado aquí sus valores ni se afirma una unidad física sin metadatos. No se ha ejecutado entrenamiento ni se han modificado arquitecturas, fórmulas de métricas, checkpoints o resultados.

La referencia identificada en TFG_AUDIT.md es Rinaldi, Gómez-Vela y Ghandehari, “Derivation of surface models using satellite imagery deep learning architectures with explainable AI”, Results in Engineering 24 (2024), 103436, DOI 10.1016/j.rineng.2024.103436. Cuando un detalle del paper no consta de forma verificable en el repositorio, se marca como pendiente de confirmar.

## 2. Datos cargados

Los dos notebooks usan las mismas rutas bajo DATA_ROOT:

| Variable | Archivo esperado | Carga | Uso |
|---|---|---|---|
| x_train | xtrain_nyc-002.npy | np.load sin parámetros de transformación | Entrada de train y validation |
| y_train | ytrain_nyc.npy | np.load sin parámetros de transformación | DSM objetivo de train y validation |
| x_test | xtest_nyc.npy | np.load sin parámetros de transformación | Entrada de test |
| y_test | ytest_nyc.npy | np.load sin parámetros de transformación | DSM objetivo de test |

No hay datasets, metadatos geoespaciales, código de generación de patches ni manifiestos de los arrays en el repositorio. El nombre “nyc” sugiere el área, pero no acredita procedencia, fecha, sensor, resolución, CRS, datum, unidad vertical ni versión.

## 3. Forma y dtype

El contrato que exige ahora la celda de auditoría es:

- x_train y x_test: NHWC, cuatro canales, por tanto (N, 128, 128, 4) si los patches conservan el tamaño usado por los modelos.
- y_train y y_test: NHWC, un canal DSM, por tanto (N, 128, 128, 1).
- Antes de crear TensorDataset, todos los arrays se convierten a float32 y se permutan a NCHW.

El número real de muestras, el dtype original guardado en cada NPY y las formas reales completas solo pueden confirmarse ejecutando la celda con Drive montado. La nueva función falla de forma explícita si no recibe cuatro canales de entrada o un canal objetivo.

## 4. Split train, validation y test

No existe un archivo independiente de validación. TensorDataset se construye con x_train/y_train y random_split reserva el 10 % para validation. La partición es reproducible dentro del notebook porque usa torch.Generator().manual_seed(SEED), actualmente SEED=42.

El test procede de archivos NPY independientes. No hay información versionada que demuestre separación espacial, ausencia de solapamiento entre patches o equivalencia con el split del paper.

La celda data-scale-audit se ejecuta después de random_split y usa exactamente train_ds.indices y val_ds.indices. Calcula por separado train, validation y test; no confunde el array fuente completo con el subconjunto real de entrenamiento.

## 5. Estadísticas añadidas

La función analyze_data_scale calcula, para cada uno de los cuatro canales de x y para el DSM y, lo siguiente:

- split, variable, canal, shape y dtype original;
- número total, valores finitos y valores no finitos;
- fracción de ceros, útil como señal inicial de posibles nodata aunque no es una prueba;
- min, max, mean y std;
- percentiles 1, 5, 25, 50, 75, 95 y 99.

Los percentiles son exactos sobre los valores finitos del split, no una estimación por minibatches. El resultado queda en el DataFrame data_scale_stats y se muestra en el notebook. No se escribe en results.csv ni se altera el manifiesto de entrenamiento.

assess_dsm_numeric_scale solo clasifica la escala numérica de forma prudente: detecta rangos compatibles con [0, 1] o con una estandarización aproximada. Un rango distinto se describe como “height-like”, pero nunca se etiqueta automáticamente como metros.

## 6. Normalización, estandarización, clipping y desnormalización

### Entrada x

No existe normalización, estandarización, clipping ni escalado explícito en los notebooks. La única transformación es:

1. carga mediante np.load;
2. conversión a float32;
3. permutación NHWC a NCHW.

No se calculan estadísticas solo sobre train para aplicar una transformación común a validation/test. Tampoco hay transformaciones por canal ni una función inversa. La llamada clip_grad_norm_ limita gradientes del modelo y no es clipping de datos.

La documentación anterior conserva como evidencia histórica un mínimo/máximo global aproximado de x_train de -3.79/16.885. Ese rango no parece reflectancia cruda en [0, 1], pero no permite deducir qué transformación previa se aplicó ni si todos los canales comparten escala.

### Objetivo y

No existe normalización, estandarización, clipping, offset ni desnormalización explícita del DSM. y se convierte a float32 y se entrega directamente al loss y a las métricas.

ARCHITECTURE_AUDIT.md conserva como evidencia histórica un rango aproximado de y_train de 0 a 405.84. Es incompatible con una normalización [0, 1] y resulta plausible para alturas urbanas, pero no prueba que la unidad sea el metro ni aclara datum, offset, edificios, terreno, nodata o valores saturados.

## 7. Escala del DSM y unidades de las métricas

MAE, RMSE y R² se calculan directamente entre pred y target procedentes del loader. No hay desnormalización previa. Por tanto:

- MAE y RMSE están en la misma unidad numérica almacenada en y_train/y_test;
- si y está en metros reales, MAE y RMSE también están en metros;
- si y fue escalado al generar los NPY, las métricas están en esa escala y no son comparables con métricas en metros;
- R² es adimensional, aunque sigue dependiendo del conjunto de píxeles y del tratamiento de nodata.

Veredicto actual: los valores históricos son compatibles con un DSM no normalizado y posiblemente expresado en una unidad de altura real, pero metros permanece pendiente de confirmar. Hace falta metadata o el script que produjo los NPY.

## 8. Nodata, máscaras y valores no finitos

No hay máscara de nodata, agua, bordes o zonas inválidas. evaluate_global incluye todos los píxeles. Tampoco filtra NaN o infinito: si aparecen, pueden contaminar loss y métricas.

La nueva tabla contabiliza valores no finitos y ceros por split/canal. Una fracción alta de ceros puede indicar suelo válido, nodata codificado como cero o ambos; no debe convertirse en máscara sin conocer el convenio del dataset.

## 9. Tabla paper_vs_notebook

| aspecto | paper | notebook actual | estado | acción necesaria |
|---|---|---|---|---|
| canales de entrada | La documentación del repositorio lo describe con imágenes radar y visible/NIR; orden y preprocesado exactos pendientes de confirmar | Cuatro canales anónimos cargados desde xtrain/xtest | Parcial | Documentar nombre, sensor, unidad, orden y generación de cada canal |
| tamaño de patch | Pendiente de confirmar | Contrato de 128 x 128 | Pendiente de equivalencia | Verificar en metodología del paper y en el generador de NPY |
| DSM objetivo | La documentación del repositorio lo describe como DSM derivado de LiDAR | Un canal y sin metadata geoespacial | Parcial | Documentar fuente, resolución, datum, unidad vertical, offset y remuestreo |
| normalización de entrada | Pendiente de confirmar | No hay transformación explícita; posible preprocesado externo desconocido | No demostrado | Recuperar preprocessing del paper y el script que creó los NPY |
| normalización de salida | Pendiente de confirmar | No existe normalización/desnormalización explícita | No demostrado | Confirmar escala del NPY y definir transformación/inversa solo si corresponde |
| split train/validation/test | Pendiente de confirmar | Test predefinido; validation aleatoria del 10 % de train con seed 42 | No equivalente aún | Confirmar split del paper y comprobar separación espacial/solapamientos |
| métricas | MAE, RMSE y R² según la documentación del repositorio | MAE, RMSE y R² globales por píxel | Parcial | Confirmar agregación por píxel o por imagen y protocolo exacto |
| unidades de MAE/RMSE | Pendiente de confirmar | Unidad numérica de y; probablemente no normalizada, metros no acreditado | Crítico pendiente | Aportar metadata vertical y, si procede, desnormalizar antes de evaluar |
| nodata/máscaras | Pendiente de confirmar | Sin máscara; todos los píxeles participan | No demostrado | Documentar convenio nodata y reproducir la máscara del paper |
| arquitectura baseline U-Net | U-Net figura como baseline; configuración exacta pendiente de confirmar | U-Net de cuatro niveles, BatchNorm, bilinear upsampling y salida lineal | Parcial | Comparar bloque por bloque, filtros, upsampling, loss e inicialización |
| arquitectura baseline U-Net++ | U-Net++ figura como baseline; configuración exacta pendiente de confirmar | Grafo nested hasta x04, sin deep supervision y salida lineal | Parcial | Confirmar deep supervision, filtros, conexiones, loss e inicialización |

## 10. Riesgos detectados

### Críticos

1. No se puede acreditar que y esté en metros; una comparación numérica de MAE/RMSE con el paper podría usar unidades distintas.
2. No se conoce el preprocesado que produjo xtrain_nyc-002.npy. Los cuatro canales pueden tener escalas heterogéneas o transformaciones previas no documentadas.
3. No se ha demostrado equivalencia del split ni ausencia de fuga espacial entre patches.
4. No se conoce el tratamiento de nodata del paper ni de los NPY.

### Medios

1. La validación aleatoria por patch puede no representar generalización espacial.
2. Las métricas del notebook son globales por píxel; el paper podría agregarlas de otra forma.
3. El rango histórico de x sugiere preprocesado externo, pero no hay parámetros para reproducirlo.
4. Los nombres de los archivos no identifican versión, hash o procedencia del dataset.

### Menores

1. Los notebooks importan herramientas de augmentación, pero no aplican augmentations en el pipeline revisado.
2. La fracción de ceros es solo diagnóstico y no distingue suelo válido de nodata.

## 11. Recomendaciones en orden

1. Ejecutar ambos notebooks hasta data-scale-audit y conservar la tabla completa de train/validation/test junto al RUN_ID.
2. Recuperar o crear una ficha de dataset con hashes de los cuatro NPY, procedencia, fechas, sensores, resolución, CRS, datum y unidad vertical.
3. Documentar explícitamente el orden y significado de los cuatro canales.
4. Recuperar el script que generó xtrain_nyc-002.npy y registrar toda normalización, log-transform, clipping o estandarización previa.
5. Confirmar si y está en metros y si contiene offset, nodata o remuestreo. No introducir una desnormalización sin esa evidencia.
6. Revisar la metodología del paper para completar las celdas pendientes de paper_vs_notebook.
7. Comparar y, si procede, reproducir el split espacial y el tratamiento de máscaras del paper.
8. Solo después decidir si hace falta normalizar x o y y actualizar evaluate_global para devolver MAE/RMSE en unidades físicas.
9. Mantener las arquitecturas y fórmulas actuales congeladas mientras se valida la equivalencia de datos.

## 12. Conclusión

El notebook no normaliza ni desnormaliza x o y de forma explícita. Las métricas se calculan directamente sobre el DSM almacenado. La evidencia histórica sugiere que y no está normalizado a [0, 1] y que podría contener alturas urbanas en una escala física, pero no permite afirmar que MAE/RMSE estén en metros.

La nueva celda convierte esta incertidumbre en evidencia reproducible por split. La comparabilidad con el paper seguirá siendo provisional hasta documentar unidades, generación de canales, split, nodata y configuraciones exactas de U-Net/U-Net++.
