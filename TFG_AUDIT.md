# Auditoría técnica del TFG

Fecha: 2026-07-29  
Repositorio: `ajrodloptfg06/TFG-DSM-DeepLearning`  
Commit revisado: `591a5ed` (`Improve attention and HRNet variants`)  
Rama: `audit/tfg-current-state`

## 1. Alcance y método

Se revisaron todos los archivos versionados: `TFG_DsmV6.ipynb`, `TFG_DsmV6_FAST_TEST.ipynb`, `ARCHITECTURE_AUDIT.md`, `NEXT_EXPERIMENTS.md` y `README`.

No hay módulos Python, tests, configuración de entorno, datasets, checkpoints ni resultados versionados. Los datos y artefactos se esperan en Google Drive.

La revisión incluyó historial Git, todas las celdas y análisis estático de arquitecturas, formas, dependencias, entrenamiento, checkpoints, resultados, métricas y escalado. No se entrenó, no se accedió a Drive y no se modificaron notebooks, checkpoints o resultados.

No pudo ejecutarse el smoke test local porque este entorno no tiene PyTorch ni `timm`. Las formas están verificadas por análisis estático. Los notebooks incluyen `sanity_check_model_shapes`, pero no conservan outputs ni contadores que prueben su ejecución.

## 2. Resumen ejecutivo

Hay dos notebooks casi idénticos. `TFG_DsmV6.ipynb` es el principal y `TFG_DsmV6_FAST_TEST.ipynb` una variante de dos épocas. Ambos contienen U-Net, U-Net++, Attention U-Net residual, una variante llamada Swin-UNet y HRNet multiescala. TransUNet no aparece.

El código actual corrige problemas de `ARCHITECTURE_AUDIT.md`: separa `last`/`best`, unifica nombres de U-Net/U-Net++, U-Net++ usa `evaluate_global`, HRNet fusiona varias escalas y una evaluación final reconstruye el CSV desde los mejores checkpoints.

Persisten riesgos críticos:

- MAIN y FAST comparten checkpoints y resultados; una prueba puede reutilizar o sobrescribir artefactos finales.
- No hay normalización explícita de `x`/`y` ni metadatos de escala y unidades.
- No hay evidencia versionada de una ejecución limpia ni de resultados.
- No se demuestra equivalencia del protocolo con el paper.

Además, conviven resultados acumulativos y regenerados; Swin/Attention no son variantes canónicas; las dependencias no tienen versiones; y el split de validación es aleatorio por patch.

Las cinco implementaciones están conectadas al entrenamiento/evaluación y por inspección aceptan `(B,4,128,128)` y producen `(B,1,128,128)`. El CSV aún no debe considerarse evidencia definitiva de comparación o equivalencia con el paper.

## 3. Notebooks y versión actual

| Notebook | Finalidad | Estado | Uso recomendado |
|---|---|---|---|
| `TFG_DsmV6.ipynb` | 50 épocas, `FAST_DEV_RUN=False`, backbones preentrenados | 37 celdas; sin outputs/contadores | Resultados finales |
| `TFG_DsmV6_FAST_TEST.ipynb` | 2 épocas, `FAST_DEV_RUN=True`, sin preentrenamiento | 38 celdas; sin outputs/contadores | Smoke test |

Ambos contienen el mismo código funcional y se actualizaron en `591a5ed`. El principal es autoritativo porque FAST remite expresamente a él para resultados finales. No hay versión interna, changelog experimental ni identificador que vincule código, datos y checkpoints.

## 4. Arquitecturas

| Arquitectura | Clase | Implementación | Instanciada | Entrenada | Evaluada desde `best` |
|---|---|---|---:|---:|---:|
| U-Net | `UNet` | Encoder-decoder de 4 niveles | Sí | Sí | Sí |
| U-Net++ | `UNetPlusPlus` | Grafo nested hasta `x04`, sin deep supervision | Sí | Sí | Sí |
| Attention U-Net | `AttentionUNet` | Gates residuales en 4 skips | Sí | Sí | Sí |
| Swin-UNet | `SwinUNet` | Swin de `timm` + decoder CNN | Sí | Sí | Sí |
| TransUNet | — | No aparece | No | No | No |
| HRNet | `HRNetRegressor` | HRNet-W18 Small + fusión multiescala | Sí | Sí | Sí |

Todas las presentes se pasan a `fit_model_resumable` y tienen evaluación de `best_*.pth`. Sin outputs/checkpoints versionados no se demuestra que se ejecutaran limpiamente.

### U-Net

Completa: 4 canales, niveles `64,128,256,512,1024`, cuatro bajadas/subidas, skips y salida lineal de un canal. No está mal conectada. No se acredita equivalencia exacta con la configuración del paper.

### U-Net++

Completa hasta `x04`, salida de un canal y pipeline común. No usa deep supervision; debe contrastarse con la variante exacta del paper.

### Attention U-Net

Completa y entrenable, pero no estándar: usa `x*(1+psi)`, amplificando skips entre `1x` y `2x`, en vez de `x*psi`. Debe llamarse `Attention-U-Net-Residual`.

### Swin-UNet

Es `swin_tiny_patch4_window7_224` como encoder más decoder convolucional, no Swin-UNet canónico con decoder Transformer/patch expanding. Etiqueta recomendada: “Swin-Tiny encoder + CNN U-Net decoder”.

El forward presupone features NHWC y siempre hace `permute(0,3,1,2)`. Sin fijar/validar `timm`, un cambio de layout puede romperlo.

### HRNet

La versión actual proyecta y fusiona todos los features, corrigiendo la versión antigua que usaba solo el último. Es razonable, aunque no necesariamente una cabeza HRNet canónica. Reescalar todo a `128x128` puede consumir bastante memoria.

### TransUNet

No hay clase, importación, entrenamiento, checkpoint ni resultado.

## 5. Contrato de formas

| Modelo | Entrada | Salida estática | Observación |
|---|---|---|---|
| U-Net | `(B,4,128,128)` | `(B,1,128,128)` | 4 bajadas/subidas |
| U-Net++ | `(B,4,128,128)` | `(B,1,128,128)` | Interpolación a referencias |
| Attention residual | `(B,4,128,128)` | `(B,1,128,128)` | 4 bajadas/subidas |
| Swin híbrido | `(B,4,128,128)` | `(B,1,128,128)` | Interpolación final explícita |
| HRNet multiescala | `(B,4,128,128)` | `(B,1,128,128)` | Features llevados a `(H,W)` |

Limitaciones:

1. no hay ejecución persistida;
2. `sanity_check_model_shapes` interpola antes de comprobar y no valida la resolución nativa;
3. `_match_prediction_to_target` también interpola silenciosamente durante train/test.

Los forwards parecen correctos, pero el test contractual debería fallar, no corregir, ante una forma distinta.

## 6. Definiciones y orden

No hay clases instanciadas antes de definirse en ejecución lineal.

Sí hay redefinición con *late binding*: `DoubleConv` se define para Attention, `AttentionUNet` referencia ese global, `DoubleConv` se redefine para U-Net y la evaluación final vuelve a instanciar Attention. Hoy ambas definiciones son equivalentes; si divergen, el modelo cargado puede construirse distinto al que creó su checkpoint.

Dependencias de estado:

- Drive y arrays deben existir antes de rutas/loaders.
- `device`, loss, métricas y entrenamiento deben definirse antes de usarse.
- `results` se inicializa tras Swin; reejecutar `append` duplica filas.
- Se reutilizan objetos y el nombre genérico `ckpt`.
- La evaluación final exige todas las definiciones previas.
- “Run all” obliga a entrenar cinco modelos; no hay selector de fase/modelo.

La dependencia del orden es alta, aunque una ejecución completa resuelve las definiciones.

## 7. Checkpoints

| Modelo | `last` | `best` |
|---|---|---|
| Swin | `last_swin_unet.pth` | `best_swin_unet.pth` |
| Attention | `last_attention_unet_residual.pth` | `best_attention_unet_residual.pth` |
| HRNet | `last_hrnet_w18_multiscale.pth` | `best_hrnet_w18_multiscale.pth` |
| U-Net | `last_unet.pth` | `best_unet.pth` |
| U-Net++ | `last_unetpp.pth` | `best_unetpp.pth` |

Los nombres son coherentes. Se reanuda desde `last` y se guarda `best` solo al mejorar RMSE.

### Colisión crítica MAIN/FAST

Ambos usan el mismo directorio, nombres y `results.csv`:

- FAST puede cargar un `last` final y no entrenar si su epoch supera 2.
- FAST puede sobrescribir `best` con dos épocas.
- MAIN puede reanudar estado FAST.
- El CSV puede sobrescribirse desde el notebook equivocado.
- No consta preentrenamiento vs. inicialización aleatoria.

Los checkpoints tampoco guardan configuración completa, `CFG`, seed/split, versiones, transformaciones, dataset, commit o modo FAST/final.

## 8. Resultados

Coexisten:

1. `results=[]` más `append` y escrituras intermedias;
2. `evaluate_best_checkpoints`, que crea filas nuevas, carga todos los `best`, recalcula test y reescribe el CSV.

Por tanto, **sí**, la celda final regenera desde los mejores checkpoints, pero antes hay una lista mutable y dependiente del orden. Si todo termina, el CSV final viene de `best`. Si se detiene, se reejecutan celdas o falta un checkpoint, puede quedar parcial/duplicado. La función final omite checkpoints ausentes y un CSV incompleto puede parecer válido.

No hay resultados versionados, `run_id`, timestamp, hash ni manifiesto.

## 9. Métricas

`evaluate_global` calcula correctamente métricas globales por píxel:

- `MAE = sum(|pred-y|)/N`;
- `RMSE = sqrt(sum((pred-y)^2)/N)`;
- `R² = 1-SS_res/SS_tot` con media global.

Usa correctamente `model.eval()` y `torch.no_grad()`.

Riesgos/matices:

- No promedia por imagen; puede diferir del paper.
- No enmascara nodata/agua/bordes.
- No detecta `NaN`/inf.
- R² guarda todos los píxeles en RAM y suma en `float32`.
- `test_loss` promedia batches sin ponderar el último; no afecta MAE/RMSE/R².
- La interpolación previa forma parte implícita del protocolo.

Las fórmulas son correctas; falta equivalencia operacional con el paper.

## 10. Normalización y escala

### `x`

Se carga, convierte a `float32`, permuta NHWC→NCHW e inspeccionan estadísticas. No se normaliza, estandariza, recorta ni transforma por canal. No se documentan nombres/unidades. No puede saberse si los `.npy` están preprocesados ni si train/test usan estadísticos comunes.

### `y`

Se convierte y usa directamente. No hay normalización ni desnormalización. Solo es correcto si ya está en las unidades finales.

### Escala frente al paper

El riesgo es alto: MAE/RMSE cambian con la unidad. Sin unidad vertical, datum, tipo de DSM, offset, clipping, remuestreo y nodata, no puede afirmarse que las cifras estén en la escala del paper.

## 11. Comparabilidad con el paper

Referencia: Rinaldi, Gómez-Vela y Ghandehari, *Derivation of surface models using satellite imagery deep learning architectures with explainable AI*, Results in Engineering 24 (2024), 103436, DOI `10.1016/j.rineng.2024.103436`.

El repositorio intenta usar cuatro canales, patches `128x128`, DSM y baselines U-Net/U-Net++. No conserva una matriz paper/notebook.

Riesgos de no equivalencia:

1. arrays sin versión/procedencia;
2. canales/preprocesado no especificados;
3. unidades DSM no acreditadas;
4. split aleatorio frente a posible split espacial;
5. posible proximidad/solapamiento de patches;
6. SmoothL1 no justificada frente al paper;
7. AdamW, coseno, 50 épocas y batch 16 sin equivalencia;
8. agregación global no contrastada;
9. sin máscara/nodata;
10. U-Net++ sin deep supervision;
11. alternativas no canónicas;
12. contaminación FAST/MAIN;
13. preentrenamiento solo para Swin/HRNet;
14. una seed y sin dispersión.

No debe atribuirse una diferencia a la arquitectura hasta verificar estos puntos.

## 12. Severidad

### Críticos

| ID | Problema | Impacto |
|---|---|---|
| C1 | MAIN/FAST comparten checkpoints y CSV | Mezcla/sobrescritura silenciosa |
| C2 | Sin unidades, transformaciones y procedencia | Escala no comparable |
| C3 | Sin trazabilidad código-datos-checkpoint-resultado | Cifras no auditables |
| C4 | Sin equivalencia demostrada con el paper | Conclusiones potencialmente inválidas |

### Medios

| ID | Problema |
|---|---|
| M1 | Resultados acumulativos y regenerados coexisten |
| M2 | Split aleatorio sin control espacial |
| M3 | Swin/Attention no canónicos |
| M4 | Dependencias sin versiones |
| M5 | Suposición NHWC rígida en Swin |
| M6 | Redefinición de `DoubleConv` |
| M7 | Smoke tests corrigen antes de validar |
| M8 | Checkpoints sin metadatos |
| M9 | Sin selector de fase/modelo |
| M10 | Loaders ignoran valores equivalentes de `CFG` |
| M11 | Agregación/máscara no contrastadas |
| M12 | TransUNet y XAI aún ausentes |

### Menores

| ID | Problema |
|---|---|
| m1 | `BASE` se redefine de `TFG_DSM` a MyDrive |
| m2 | Imports repetidos y celda vacía |
| m3 | `ckpt` reutilizado |
| m4 | README mínimo y mojibake |
| m5 | Visualización aleatoria |
| m6 | `test_loss` no pondera el último batch |

## 13. Recomendaciones, en orden

### Fase 1 — Aislar experimentos

1. Separar MAIN/FAST con `RUN_MODE`, `RUN_ID` y subdirectorios.
2. Impedir que FAST escriba resultados finales.
3. Guardar manifiesto: commit, modelo, seed, hiperparámetros, versiones, datos y estadísticos.
4. Fijar versiones compatibles con Colab.

### Fase 2 — Datos, escala y paper

5. Documentar orden, unidades y generación de canales.
6. Calcular estadísticas solo en train y registrar transformaciones.
7. Documentar unidad, datum, offset, nodata y escala de `y`; centralizar desnormalización.
8. Crear tabla “paper vs reproducción”.
9. Revisar split espacial y persistir índices.

### Fase 3 — Evaluación

10. Mantener una sola evaluación oficial desde `best`, sin borrar resultados existentes.
11. Fallar claramente ante checkpoints ausentes/incompatibles.
12. Añadir `run_id`, hash, epoch, seed y modo al CSV.
13. Validar formas sin interpolación.
14. Probar métricas con casos pequeños y documentar agregación/máscara.

### Fase 4 — Notebook reproducible

15. Separar configuración, datos, modelos, smoke tests, train, evaluación y visualización.
16. Evitar redefinir `DoubleConv`.
17. Usar `CFG["batch_size"]` y `CFG["num_workers"]`.
18. Seleccionar modelo y modo `smoke/evaluate/train`.
19. Fijar muestras comunes para visualización.

### Fase 5 — Arquitecturas

20. Consolidar U-Net/U-Net++ como baselines.
21. Renombrar variantes o implementar versiones canónicas.
22. Añadir Attention U-Net estándar como control.
23. Decidir entre Swin híbrido y Swin-UNet canónico.
24. Añadir TransUNet tras estabilizar el pipeline.
25. Añadir XAI comparable.
26. Repetir con varias seeds y reportar dispersión.

## 14. Veredicto por requisito

| Requisito | Veredicto |
|---|---|
| Notebooks | Dos; MAIN referencia, FAST prueba |
| U-Net | Implementada, entrenada y evaluada |
| U-Net++ | Implementada, entrenada y evaluada; sin deep supervision |
| Attention U-Net | Variante residual implementada, entrenada y evaluada |
| Swin-UNet | Híbrido Swin+CNN implementado, entrenado y evaluado |
| TransUNet | No aparece |
| HRNet | Multiescala implementada, entrenada y evaluada |
| Formas | Correctas estáticamente; ejecución no demostrada |
| Clases antes de definir | No; redefinición frágil de `DoubleConv` |
| Dependencia de orden | Alta |
| Checkpoints | Coherentes; colisionan MAIN/FAST |
| Resultados desde `best` | Sí al final; coexiste flujo acumulativo |
| MAE/RMSE/R² | Correctas como métricas globales por píxel |
| Normalización/desnormalización | No existe explícitamente |
| Riesgo de escala distinta | Alto |
| Riesgo de no equivalencia | Alto |

## 15. Conclusión

El proyecto está más sólido que en la auditoría anterior: las cinco arquitecturas presentes comparten pipeline, `best` se separa de `last` y existe regeneración final.

El cuello de botella no es añadir otra arquitectura, sino asegurar identidad experimental: datos/unidades conocidos, runs aislados, checkpoints trazables, evaluación única y equivalencia documentada con el paper. Resolver C1–C4 hará defendibles las comparaciones posteriores.
