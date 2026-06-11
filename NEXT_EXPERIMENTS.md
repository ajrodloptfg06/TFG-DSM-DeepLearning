# Next experiments

Ideas pendientes para mejorar o auditar despues de los cambios actuales:

1. Comparar U-Net y U-Net++ como baseline fijo antes de relanzar todos los modelos.
2. Revisar normalizacion por canal de entrada y escala del DSM objetivo.
3. Probar perdidas alternativas para valores altos de DSM: `MSELoss`, `L1Loss`, `SmoothL1Loss`, o mezcla MAE+RMSE.
4. Revisar split espacial train/validation/test para reducir la brecha entre `val_R2` y `test_R2`.
5. Swin-UNet: la version actual es un Swin encoder con decoder simple; si se mantiene, conviene probar una rama CNN de alta resolucion, decoder mas fuerte, y learning rates separados para encoder/decoder.
6. Visualizacion: revisar varias muestras fijas, no solo una aleatoria, y reportar MAE/RMSE por muestra.
7. Si se cambia la normalizacion del objetivo, desnormalizar siempre `pred` y `target` antes de MAE/RMSE/R2.
