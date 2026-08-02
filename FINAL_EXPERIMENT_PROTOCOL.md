# Protocolo del experimento final

Fecha de revisión: 2026-08-03

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

### 1.3 Veredicto de preparación

**El aislamiento FAST/FINAL y la evaluación trazable están listos. El notebook todavía no se considera totalmente listo para una ejecución FINAL desatendida de arriba abajo.**

No hay un error que mezcle automáticamente FAST y FINAL, pero antes de gastar el tiempo de entrenamiento conviene resolver o controlar estos puntos:

1. Un RUN_ID explícito reutilizado reanuda checkpoints anteriores y puede conservar campos antiguos del manifiesto.
2. Swin, Attention y HRNet se evalúan individualmente sobre test antes de la evaluación oficial.
3. Quedan varios modelos residentes en GPU durante la ejecución secuencial.
4. El sanity check conjunto aparece después de entrenar los tres modelos alternativos.
5. El sanity check interpola una salida espacial incorrecta antes de comprobarla y podría ocultar un incumplimiento nativo.
6. PyTorch y timm se instalan sin versión fijada.
7. El seed se establece globalmente una vez, pero no se reinicia antes de cada modelo; el orden de ejecución puede afectar inicialización y shuffle.
8. El commit puede quedar como null en run_manifest.json si Colab abrió el notebook desde GitHub sin un checkout Git ni GIT_COMMIT.

Estos puntos se documentan en Cambios recomendados y no se modifican en esta rama.

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
13. Ejecutar el análisis completo de escala para train, validation y test.
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

En el notebook actual, las definiciones y el sanity check conjunto no están ordenados antes de todos los entrenamientos. Hasta aplicar el cambio recomendado, deben respetarse los smoke tests individuales y evitar convertir la ejecución manual fuera de orden en el run oficial.

### 6.4 Entrenamiento

23. Entrenar los cinco modelos dentro del mismo RUN_ID.
24. Verificar para cada modelo que se actualizan last y best en CKPT_DIR.
25. No copiar checkpoints de otra carpeta ni renombrar artefactos manualmente.
26. Si Colab se interrumpe, reanudar solo con el mismo RUN_ID y después de comprobar que manifiesto, configuración, datos y commit siguen siendo idénticos.
27. No consultar test para decidir hiperparámetros, épocas, arquitectura o selección de checkpoint.
28. Liberar la memoria del modelo anterior antes de pasar al siguiente si la GPU se acerca al límite.

### 6.5 Evaluación oficial

29. Comprobar que existen los cinco best checkpoints.
30. Ejecutar una sola vez evaluate_best_checkpoints.
31. Confirmar que cada fila usa el checkpoint best del mismo RUN_ID.
32. Guardar results.csv en results/final/RUN_ID.
33. Confirmar que run_manifest.json se actualiza con los checkpoints y la ruta del CSV.
34. Tratar el CSV regenerado por esa función como única tabla oficial.
35. Ejecutar visualizaciones solo después de congelar el CSV oficial; no usarlas para elegir el ganador.

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

- Reutilizar RUN_ID puede reanudar pesos antiguos y conservar metadatos anteriores.
- Interrumpir un run reutilizado puede dejar un results.csv antiguo junto a checkpoints parcialmente actualizados.
- Las evaluaciones intermedias de Swin, Attention y HRNet exponen test antes de la evaluación oficial.
- Mantener varios modelos en GPU puede provocar OOM.
- El sanity check conjunto se ejecuta demasiado tarde y puede corregir shapes por interpolación.
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

## 13. Cambios recomendados

No se aplican en esta rama.

### Prioridad alta antes del run costoso

1. **Centralizar definiciones y sanity check antes de cualquier entrenamiento.** El shape check conjunto debería ejecutarse con los cinco modelos ya definidos y antes de entrenarlos.
2. **Hacer estricto el shape check.** Debe comprobar la salida nativa y fallar antes de interpolar.
3. **Eliminar o saltar las evaluaciones intermedias sobre test.** Test debe consultarse una vez, mediante evaluate_best_checkpoints, cuando las decisiones estén congeladas.
4. **Liberar memoria entre modelos.** Eliminar referencias, mover el modelo anterior a CPU o usar una función secuencial que no retenga cinco redes en GPU.
5. **Impedir la reutilización accidental de un RUN_ID final.** Para un run limpio, fallar si las carpetas ya contienen artefactos; habilitar resume solo mediante una opción explícita.
6. **Validar compatibilidad al reanudar.** Comparar commit, modelo, seed, épocas, batch, learning rate, weight decay y rutas/fingerprints de datos con el manifiesto existente.

### Prioridad media para reproducibilidad

7. Fijar versiones de PyTorch, timm y dependencias.
8. Registrar el SHA mediante GIT_COMMIT cuando Colab no sea un checkout Git.
9. Guardar índices o fingerprints del split.
10. Decidir y documentar una política de reseeding por modelo y del generador del DataLoader.
11. Añadir fingerprints de los cuatro arrays o de sus metadatos.
12. Planificar varias seeds después del primer run final reproducible.

Estos cambios afectan al orden y la seguridad del pipeline, no a las arquitecturas, métricas, loss, optimizador ni scheduler.

## 14. Conclusión operativa

El notebook ya separa correctamente FAST y FINAL, usa checkpoints last/best por modelo y regenera una tabla oficial desde los cinco mejores checkpoints. Sin embargo, antes de iniciar un entrenamiento final costoso se recomienda corregir el orden del sanity check, eliminar las consultas intermedias a test, controlar memoria y hacer explícita la política de RUN_ID/resume.

Hasta entonces, el notebook está **preparado a nivel de aislamiento y trazabilidad, pero condicionado para una ejecución final completamente limpia y desatendida**.
