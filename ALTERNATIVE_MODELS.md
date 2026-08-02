# Revisión de modelos alternativos

Fecha: 2026-08-03

Rama: **alternative-models-review**

## 1. Alcance y método

Este documento revisa **AttentionUNet**, **SwinUNet** y **HRNetRegressor** en **TFG_DsmV6.ipynb** y **TFG_DsmV6_FAST_TEST.ipynb**. Las definiciones son iguales en ambos notebooks.

La revisión es arquitectónica y estática. No se han modificado modelos, entrenamiento ni métricas; no se han ejecutado entrenamientos; y no se ha implementado TransUNet. El entorno local no dispone de PyTorch ni timm, por lo que las formas se han comprobado siguiendo cada forward. Los notebooks contienen smoke tests, pero no conservan resultados ni contadores de ejecución.

## 2. Veredicto ejecutivo

Los tres modelos están conectados para regresión DSM y, por construcción, cumplen:

**(B, 4, 128, 128) → (B, 1, 128, 128)**

En todos, la última operación aprendible es una convolución que produce un canal y no se aplica sigmoid, softmax, ReLU ni otra activación a la predicción. La salida es lineal y representa una variable continua.

No se ha encontrado un error arquitectónico evidente que justifique modificar los notebooks. Sí deben declararse estas limitaciones:

- Attention U-Net usa atención residual **x × (1 + psi)**, no el filtrado estándar **x × psi**.
- SwinUNet es un híbrido con encoder Swin-Tiny y decoder CNN, no una reproducción demostrada del Swin-UNet canónico.
- HRNetRegressor usa HRNet-W18-Small como extractor y añade una cabeza multiescala propia.
- La versión de timm no está fijada, por lo que layouts, canales y adaptación de pesos deben comprobarse en Colab.

## 3. Contrato de entrada, salida y regresión

| Modelo | Entrada de cuatro canales | Recuperación de 128 × 128 | Cabeza de un canal | Activación final | Resultado estático |
|---|---|---|---|---|---|
| Attention U-Net residual | in_ch=4 | Cuatro subidas: 8 → 16 → 32 → 64 → 128 | Conv2d(64, 1, 1) | Ninguna | (B,1,128,128) |
| Swin híbrido | in_chans=4 en timm | Interpolación final explícita | Conv2d(32, 1, 1) | Ninguna | (B,1,128,128) |
| HRNet multiescala | in_chans=4 en timm | Cada feature se interpola a (H,W) | Conv2d(64, 1, 1) | Ninguna | (B,1,128,128) |

El Sigmoid de AttentionGate es interno: genera el mapa de atención y no limita el DSM. No hay sigmoid ni softmax después de la cabeza de regresión.

## 4. Attention U-Net residual

### 4.1 Estructura encoder-decoder

AttentionUNet usa por defecto **in_ch=4**, **base_ch=64**, **up_mode=bilinear** y **residual_attention=True**. Sus bloques DoubleConv realizan dos secuencias Conv2d 3 × 3 → BatchNorm → ReLU, conservando la resolución.

El encoder tiene cuatro max-pools y un bottleneck. El decoder tiene cuatro subidas. Cada salida del decoder actúa como señal de gating para filtrar el skip del mismo nivel; luego se concatenan decoder y skip atendido y se fusionan con otro DoubleConv.

### 4.2 Canales y resoluciones

| Etapa | Operación | Canales de salida | Resolución para patch 128 |
|---|---|---:|---:|
| e1 | DoubleConv(4,64) | 64 | 128 × 128 |
| e2 | pool + DoubleConv(64,128) | 128 | 64 × 64 |
| e3 | pool + DoubleConv(128,256) | 256 | 32 × 32 |
| e4 | pool + DoubleConv(256,512) | 512 | 16 × 16 |
| b | pool + DoubleConv(512,1024) | 1024 | 8 × 8 |
| d4 | subida + gate e4 + fusión | 512 | 16 × 16 |
| d3 | subida + gate e3 + fusión | 256 | 32 × 32 |
| d2 | subida + gate e2 + fusión | 128 | 64 × 64 |
| d1 | subida + gate e1 + fusión | 64 | 128 × 128 |
| salida | convolución 1 × 1 | 1 | 128 × 128 |

La subida bilineal incluye una proyección 1 × 1, BatchNorm y ReLU para reducir canales. Existe up_mode=transpose, aunque las instanciaciones actuales usan el modo bilineal por defecto.

### 4.3 Skip connections y AttentionGate

Cada gate recibe **x**, la feature del encoder, y **g**, la feature del decoder. Si sus resoluciones no coinciden, g se interpola a la de x. Después:

1. W_g y W_x proyectan ambos tensores a F_int canales con convoluciones 1 × 1 y BatchNorm.
2. Se suman las proyecciones y se aplica ReLU.
3. Otra convolución 1 × 1 reduce a un mapa espacial de un canal.
4. Un Sigmoid produce psi en el intervalo (0,1).
5. El mapa se aplica al skip por broadcasting sobre todos sus canales.

| Gate | Canales skip / gating | Canales intermedios |
|---|---:|---:|
| att4 | 512 / 512 | 256 |
| att3 | 256 / 256 | 128 |
| att2 | 128 / 128 | 64 |
| att1 | 64 / 64 | 32 |

### 4.4 Atención residual x × (1 + psi)

Con residual_attention=True, el gate devuelve **x_att = x × (1 + psi)**. Como psi está entre 0 y 1, el factor está entre 1 y 2: las posiciones con atención baja se conservan aproximadamente a 1× y las de atención alta se amplifican hasta aproximadamente 2×.

Esta elección protege detalles finos del encoder, potencialmente relevantes para bordes y cambios de altura. Como contrapartida, no puede suprimir directamente información irrelevante y puede amplificar activaciones o ruido. Su eficacia solo puede determinarse con resultados controlados.

Con residual_attention=False se usaría x × psi, pero ese no es el flujo de las instanciaciones actuales.

### 4.5 Diferencia frente a U-Net clásica

La profundidad, los canales y el patrón encoder-decoder son comparables a la U-Net baseline. La diferencia es que cada skip pasa por un gate condicionado por el decoder antes de concatenarse. Además, el gate es residual: conserva y potencialmente amplifica el skip en vez de atenuarlo.

## 5. Swin híbrido con decoder CNN

### 5.1 Uso de timm.create_model

SwinUNet crea el encoder Swin-Tiny con:

- modelo: **swin_tiny_patch4_window7_224**;
- **features_only=True**, para obtener una pirámide de features;
- **in_chans=4**, para adaptar el patch embedding a cuatro canales;
- **img_size=128**, para los patches del proyecto;
- pretrained=False en FAST y pretrained=True en FINAL mediante la configuración central.

Cuando se solicitan pesos preentrenados con cuatro canales, la adaptación de la primera proyección depende de timm y debe registrarse junto con su versión.

### 5.2 Encoder Swin Transformer

Swin-Tiny aplica patch embedding de tamaño 4 y cuatro etapas jerárquicas con self-attention por ventanas desplazadas. Para 128 × 128, la configuración esperada es:

| Feature | Layout antes de convertir | Canales | Resolución esperada |
|---|---|---:|---:|
| f1 | NHWC | 96 | 32 × 32 |
| f2 | NHWC | 192 | 16 × 16 |
| f3 | NHWC | 384 | 8 × 8 |
| f4 | NHWC | 768 | 4 × 4 |

El código obtiene los canales mediante encoder.feature_info.channels(), pero presupone exactamente cuatro features.

### 5.3 Conversión NHWC → NCHW

La implementación seleccionada de Swin en timm produce features NHWC. El notebook convierte cada tensor mediante **permute(0, 3, 1, 2)** para que las convoluciones reciban NCHW.

La conversión es coherente con el backbone seleccionado, pero incondicional. Sin una versión fijada de timm, un cambio de API, layout o backbone podría convertir erróneamente un tensor ya NCHW. En Colab debe validarse cada eje contra feature_info.channels() antes de entrenar.

### 5.4 Bottleneck y decoder CNN

El último feature f4 pasa por dos bloques ConvBNAct: **768 → 256 → 256**. Después, tres DecoderBlock interpolan a la resolución del skip, concatenan y aplican dos convoluciones:

| Etapa | Entrada profunda | Skip | Salida | Resolución |
|---|---:|---:|---:|---:|
| dec3 | 256 | f3 (384) | 256 | 8 × 8 |
| dec2 | 256 | f2 (192) | 128 | 16 × 16 |
| dec1 | 128 | f1 (96) | 64 | 32 × 32 |
| head | 64 | — | 1 | 32 × 32 |
| salida | interpolación bilineal | — | 1 | 128 × 128 |

La cabeza es ConvBNAct(64,32) seguida de Conv2d(32,1,1). La ampliación final 32 → 128 es explícita y no aprendible.

### 5.5 Por qué debe llamarse Swin híbrido

El encoder sí es un Swin Transformer jerárquico. El decoder usa interpolación bilineal, concatenaciones y convoluciones CNN; no hay etapas Transformer simétricas ni patch expanding propios de algunas formulaciones canónicas.

La denominación recomendada en memoria y tablas es **Swin-Tiny encoder + decoder CNN tipo U-Net**, o **Swin híbrido**. Puede conservarse SwinUNet como nombre de clase por compatibilidad si se explica el matiz.

## 6. HRNet multiescala

### 6.1 Backbone HRNet-W18-Small

HRNetRegressor usa timm.create_model con **hrnet_w18_small**, **features_only=True** e **in_chans=4**.

HRNet mantiene ramas de distinta resolución e intercambia información entre ellas. En W18-Small, las ramas internas finales tienen canales 16, 32, 64 y 128. El wrapper features_only de timm usa por defecto features incrementadas y añade el stem; el código espera una lista de cinco mapas y consulta sus canales con feature_info.channels().

Para el comportamiento actual de timm, la pirámide esperada para 128 × 128 es:

| Feature extraído | Canales informados | Resolución esperada |
|---|---:|---:|
| stem | 64 | 64 × 64 |
| rama incrementada 1 | 128 | 32 × 32 |
| rama incrementada 2 | 256 | 16 × 16 |
| rama incrementada 3 | 512 | 8 × 8 |
| rama incrementada 4 | 1024 | 4 × 4 |

Estos valores dependen de la versión y opciones de timm; el código no los fija manualmente.

### 6.2 Normalización del layout

El método _to_nchw usa el canal esperado de feature_info:

- si shape[1] coincide con expected_ch, conserva NCHW;
- si shape[-1] coincide, convierte NHWC a NCHW;
- en otro caso, lanza ValueError.

Esta comprobación es más robusta que asumir un layout.

### 6.3 Proyección, reescalado y concatenación

Cada mapa se procesa así:

1. Se normaliza a NCHW.
2. HRFuseBlock aplica Conv2d 1 × 1 → BatchNorm → ReLU y proyecta a fusion_ch=64.
3. Se interpola bilinealmente a 128 × 128.
4. Todos los mapas se concatenan por canales.

Con cinco features, la concatenación tiene **5 × 64 = 320 canales** a resolución completa.

### 6.4 Head convolucional final

La cabeza aplica:

- Conv2d(320,128,3,padding=1) → BatchNorm → ReLU;
- Conv2d(128,64,3,padding=1) → BatchNorm → ReLU;
- Conv2d(64,1,1).

El padding conserva 128 × 128 y la última convolución produce el DSM lineal. La cabeza aprovecha todas las escalas, pero no es una cabeza canónica de HRNet. Llevar cinco mapas a resolución completa también puede consumir bastante memoria.

## 7. Comparación de los tres modelos

| Aspecto | Attention U-Net residual | Swin híbrido | HRNet multiescala |
|---|---|---|---|
| Tipo de arquitectura | CNN encoder-decoder con gates | Encoder Transformer jerárquico + decoder CNN | Backbone CNN multirama + cabeza propia |
| Cómo captura contexto | Bottleneck y gating del decoder | Self-attention por ventanas y jerarquía Swin | Intercambio entre ramas de varias escalas |
| Cómo conserva detalle espacial | Skips modulados residualmente | Skips jerárquicos; salida final interpolada | Ramas de alta resolución y fusión de escalas |
| Diferencia frente a U-Net | Gates en los cuatro skips | Encoder Swin y tres etapas de decoder | Extracción paralela multirresolución, sin decoder U-Net |
| Posible interés para DSM | Realzar bordes o estructuras sin borrar el skip | Combinar contexto jerárquico y reconstrucción CNN | Integrar contexto conservando alta resolución |
| Posibles limitaciones | No suprime; puede amplificar ruido | No canónico; depende de layout/versión; interpolación final 4× | Cabeza no canónica; mayor uso de memoria |

Las posibles ventajas son hipótesis arquitectónicas, no resultados ni evidencia de mejora frente a U-Net o U-Net++.

## 8. Limitaciones de denominación

### Attention U-Net

Debe denominarse **Attention U-Net residual** o **Attention-U-Net-Residual**: el gate por defecto usa x × (1 + psi), mientras que una atención multiplicativa convencional usaría x × psi.

### Swin-UNet

SwinUNet implementa un **Swin-Tiny encoder + decoder CNN tipo U-Net**. No se ha demostrado que coincida con un Swin-UNet canónico. El nombre corto puede mantenerse en código, pero la memoria y las tablas deben usar una etiqueta inequívoca.

### HRNet

HRNetRegressor usa **HRNet-W18-Small como backbone con una cabeza multiescala propia**. Proyectar, ampliar y concatenar todas las features es una decisión del proyecto, no una cabeza HRNet estándar demostrada.

## 9. Problemas y riesgos arquitectónicos encontrados

No se ha identificado un fallo estático de canales o resoluciones en las configuraciones concretas. Sí existen estos riesgos:

1. **Dependencias sin fijar:** no hay archivo de entorno ni versión de timm; afecta layouts y canales.
2. **Supuesto rígido en Swin:** espera cuatro features NHWC y siempre las permuta. Es correcto para el modelo seleccionado en la implementación actual de timm, pero no es genérico.
3. **Smoke test no persistido:** las celdas existen, pero no hay outputs versionados de una ejecución limpia.
4. **Attention residual no supresiva:** x × (1 + psi) conserva o amplifica; debe describirse como residual y validarse.
5. **Coste de HRNet:** concatenar cinco features de 64 canales a 128 × 128 crea un tensor de 320 canales y puede elevar la memoria.

Ningún riesgo exige cambiar la arquitectura dentro del alcance de esta rama.

## 10. Cómo defender estos modelos ante el tutor

> Los tres modelos alternativos representan mecanismos distintos para estimar DSM: atención residual sobre skips, contexto jerárquico con un encoder Transformer y mantenimiento explícito de múltiples resoluciones. Todos reciben los mismos cuatro canales y generan un mapa continuo de un canal con igual resolución. Se compararán bajo el mismo pipeline frente a U-Net y U-Net++, pero no se afirmará superioridad hasta evaluar los mejores checkpoints con datos, splits y métricas equivalentes.

- **Attention U-Net residual:** investiga si un gate condicionado por el decoder realza información útil sin eliminar detalles espaciales.
- **Swin híbrido:** investiga un encoder de atención por ventanas combinado con un decoder CNN práctico; no se presenta como Swin-UNet canónico.
- **HRNet multiescala:** investiga la fusión de features de alta y baja resolución con una cabeza diseñada para regresión densa.

Conviene separar tres niveles de evidencia:

1. **Verificado en código:** canales, conexiones, tratamiento de layouts y salida lineal.
2. **Pendiente de smoke test reproducible:** ejecución con las versiones registradas de PyTorch y timm.
3. **Pendiente de experimentación final:** rendimiento, estabilidad, memoria y comparación con baselines.

## 11. Comprobaciones pendientes antes de los runs finales

1. Registrar y, preferiblemente, fijar las versiones de PyTorch y timm usadas en Colab.
2. Ejecutar un smoke test nativo y guardar (B,4,128,128) → (B,1,128,128) sin corregir la salida con el helper de entrenamiento.
3. Para Swin, imprimir feature_info.channels(), reducciones y shapes antes/después de NHWC → NCHW.
4. Para HRNet, imprimir número de features, canales, reducciones y shapes de la versión concreta de timm.
5. Registrar si FINAL usa pesos preentrenados y cómo timm adapta tres canales a cuatro.
6. Mantener idénticos dataset, split, loss, optimizador, épocas, seed y evaluación frente a U-Net y U-Net++.

## 12. Referencias técnicas

- Extracción de features en timm: <https://huggingface.co/docs/timm/en/feature_extraction>
- Swin en timm: <https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py>
- HRNet en timm: <https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/hrnet.py>

La conducta reproducible depende de la versión concreta registrada en run_manifest.json.

## 13. Conclusión

Las tres alternativas están conectadas coherentemente para regresión DSM y cumplen estáticamente el contrato de formas. Attention U-Net es residual; Swin es un híbrido encoder Transformer-decoder CNN; y HRNet emplea una fusión multiescala propia. No se han producido ni inventado resultados, y no hay base todavía para afirmar que alguna supere a U-Net o U-Net++.
