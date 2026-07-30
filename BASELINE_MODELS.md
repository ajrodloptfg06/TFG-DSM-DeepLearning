# Revisión de modelos baseline: U-Net y U-Net++

Fecha: 2026-07-30

Rama: `baseline-models-review`

## 1. Alcance

Este documento revisa `UNet` y `UNetPlusPlus` en `TFG_DsmV6.ipynb` y `TFG_DsmV6_FAST_TEST.ipynb`. La revisión es arquitectónica y estática: no se han modificado modelos, entrenamiento ni métricas, y no se ha ejecutado entrenamiento. El entorno local no dispone de PyTorch, por lo que el contrato de formas se ha comprobado siguiendo las operaciones de cada `forward`.

## 2. Veredicto ejecutivo

Las dos implementaciones son coherentes como baselines de regresión DSM:

- reciben cuatro canales;
- conservan la resolución espacial de `128 x 128`;
- producen un único canal;
- terminan en una convolución `1 x 1`;
- no aplican `sigmoid`, `softmax`, `ReLU` ni otra activación a la predicción;
- devuelven, por construcción, `(B, 1, 128, 128)` para una entrada `(B, 4, 128, 128)`.

No se ha encontrado un error arquitectónico evidente que justifique cambiar el código en esta fase.

## 3. Bloque convolucional común

U-Net usa `DoubleConv` y U-Net++ usa `ConvBlock`. Ambos implementan la misma secuencia:

1. convolución `3 x 3`, `padding=1`, sin bias;
2. `BatchNorm2d`;
3. `ReLU`;
4. segunda convolución `3 x 3`, `padding=1`, sin bias;
5. `BatchNorm2d`;
6. `ReLU`.

El `padding=1` conserva altura y anchura dentro de cada bloque. La reducción espacial ocurre únicamente mediante `MaxPool2d(2)`.

## 4. U-Net

### 4.1 Estructura

`UNet` usa por defecto `in_ch=4`, `base_ch=64`, `bilinear=True` y `out_ch=1`. El encoder contiene el bloque inicial y cuatro niveles `Down`. Cada `Down` aplica `MaxPool2d(2)` y después un `DoubleConv`: la resolución se divide por dos y los canales se duplican.

El decoder contiene cuatro niveles `Up`. Con la configuración usada:

1. el tensor profundo se amplía por interpolación bilineal;
2. se ajusta exactamente a la resolución del skip si fuera necesario;
3. se concatena con el skip del encoder por el eje de canales;
4. un `DoubleConv` fusiona la concatenación.

Las conexiones skip son `x4 -> up1`, `x3 -> up2`, `x2 -> up3` y `x1 -> up4`. La cabeza `Conv2d(64, 1, kernel_size=1)` proyecta la última feature map al DSM.

### 4.2 Canales y resoluciones

| Etapa | Operación | Entrada | Salida | Resolución para patch 128 |
|---|---|---:|---:|---:|
| `x1` | `DoubleConv` | 4 | 64 | `128 x 128` |
| `x2` | max-pool + `DoubleConv` | 64 | 128 | `64 x 64` |
| `x3` | max-pool + `DoubleConv` | 128 | 256 | `32 x 32` |
| `x4` | max-pool + `DoubleConv` | 256 | 512 | `16 x 16` |
| `x5` | max-pool + `DoubleConv` | 512 | 1024 | `8 x 8` |
| `up1` | upsample + skip `x4` | 1024 + 512 | 512 | `16 x 16` |
| `up2` | upsample + skip `x3` | 512 + 256 | 256 | `32 x 32` |
| `up3` | upsample + skip `x2` | 256 + 128 | 128 | `64 x 64` |
| `up4` | upsample + skip `x1` | 128 + 64 | 64 | `128 x 128` |
| salida | convolución `1 x 1` | 64 | 1 | `128 x 128` |

### 4.3 Contrato de forma

Para `x.shape = (B, 4, 128, 128)`, cuatro max-pools generan resoluciones 64, 32, 16 y 8; cuatro upsamplings recuperan 16, 32, 64 y 128; las convoluciones con padding conservan cada resolución; y la cabeza `1 x 1` solo cambia canales. Resultado nativo: `(B, 1, 128, 128)`.

El código permite convolución transpuesta con `bilinear=False`, pero las instanciaciones actuales usan el valor por defecto bilineal.

## 5. U-Net++

### 5.1 Estructura general

`UNetPlusPlus` usa `in_ch=4`, `base_ch=64` y `out_ch=1`. Mantiene cinco niveles de resolución, como U-Net, pero reemplaza cada skip directo por un camino denso y anidado.

En `xij`, `i` representa la profundidad o nivel espacial y `j` la etapa dentro del camino skip anidado. Los nodos del encoder son `x00`, `x10`, `x20`, `x30` y `x40`. Los nodos con `j > 0` fusionan los nodos anteriores de su fila y una feature ampliada de la fila inferior.

### 5.2 Mapa de nodos

| Nivel | Resolución | Canales de salida | Nodos presentes |
|---:|---:|---:|---|
| 0 | `128 x 128` | 64 | `x00`, `x01`, `x02`, `x03`, `x04` |
| 1 | `64 x 64` | 128 | `x10`, `x11`, `x12`, `x13` |
| 2 | `32 x 32` | 256 | `x20`, `x21`, `x22` |
| 3 | `16 x 16` | 512 | `x30`, `x31` |
| 4 | `8 x 8` | 1024 | `x40` |

### 5.3 Conexiones anidadas

Encoder:

- `x00 = conv00(x)`;
- `x10 = conv10(pool(x00))`;
- `x20 = conv20(pool(x10))`;
- `x30 = conv30(pool(x20))`;
- `x40 = conv40(pool(x30))`.

Primera etapa: `x01 <- [x00, up(x10)]`, `x11 <- [x10, up(x20)]`, `x21 <- [x20, up(x30)]` y `x31 <- [x30, up(x40)]`.

Segunda etapa: `x02 <- [x00, x01, up(x11)]`, `x12 <- [x10, x11, up(x21)]` y `x22 <- [x20, x21, up(x31)]`.

Tercera etapa: `x03 <- [x00, x01, x02, up(x12)]` y `x13 <- [x10, x11, x12, up(x22)]`.

Etapa final: `x04 <- [x00, x01, x02, x03, up(x13)]`.

Todas las ampliaciones usan interpolación bilineal al tamaño exacto del nodo de referencia.

### 5.4 Canales de concatenación

| Nodo | Canales concatenados | Canales de salida |
|---|---:|---:|
| `x01` | `64 + 128 = 192` | 64 |
| `x11` | `128 + 256 = 384` | 128 |
| `x21` | `256 + 512 = 768` | 256 |
| `x31` | `512 + 1024 = 1536` | 512 |
| `x02` | `64 + 64 + 128 = 256` | 64 |
| `x12` | `128 + 128 + 256 = 512` | 128 |
| `x22` | `256 + 256 + 512 = 1024` | 256 |
| `x03` | `64 + 64 + 64 + 128 = 320` | 64 |
| `x13` | `128 + 128 + 128 + 256 = 640` | 128 |
| `x04` | `64 + 64 + 64 + 64 + 128 = 384` | 64 |

Las dimensiones declaradas en cada `ConvBlock` coinciden con estas concatenaciones.

### 5.5 Salida y forma

La predicción es exclusivamente `outc(x04)`, con `outc = Conv2d(64, 1, kernel_size=1)`. `x04` tiene la resolución de `x00` y de la entrada. Por tanto, `(B, 4, 128, 128) -> (B, 1, 128, 128)` de forma nativa.

## 6. Deep supervision en U-Net++

La implementación actual **no tiene deep supervision**:

- solo existe una cabeza `outc`;
- se aplica únicamente a `x04`;
- no hay cabezas auxiliares sobre `x01`, `x02` o `x03`;
- `forward` devuelve un tensor, no una lista de predicciones;
- el loss recibe una sola predicción.

El modelo conserva el grafo de skips densos y anidados de U-Net++. Deep supervision es una variante opcional, pero no debe atribuirse a esta implementación. Falta confirmar si la U-Net++ del paper la utilizó.

## 7. Adecuación para regresión DSM

| Requisito | U-Net | U-Net++ |
|---|---|---|
| Cuatro canales de entrada | Sí | Sí |
| Un canal de salida | Sí | Sí |
| Salida `128 x 128` | Sí | Sí |
| `sigmoid` final | No | No |
| `softmax` final | No | No |
| Otra activación final | No | No |
| Salida lineal | Sí | Sí |

La salida lineal es apropiada para una variable continua sin restringirla a `[0, 1]` o a clases. Puede producir valores negativos; no es un fallo estructural. Su tratamiento debe decidirse conociendo escala y datum del DSM, no añadiendo una activación sin evidencia.

## 8. Diferencias entre U-Net y U-Net++

| Aspecto | U-Net | U-Net++ |
|---|---|---|
| Encoder | Cinco niveles | Cinco niveles |
| Skip connections | Una conexión directa por nivel | Caminos densos y anidados |
| Decoder | Cuatro bloques secuenciales | Malla de nodos `xij` |
| Upsampling | Bilineal por defecto; transposed conv disponible | Bilineal al tamaño de referencia |
| Fusión | Skip + decoder | Nodos previos + nivel inferior |
| Salida | Último bloque del decoder | Nodo `x04` |
| Deep supervision | No aplica | No implementada |
| Complejidad | Menor | Mayor por concatenaciones y bloques anidados |

U-Net es más sencilla e interpretable. U-Net++ intenta reducir la brecha semántica entre encoder y decoder refinando progresivamente los skips, a costa de más cómputo y memoria.

## 9. Comparabilidad con los baselines del paper

La documentación del repositorio permite afirmar que el paper incluye U-Net y U-Net++ como baselines. No permite afirmar que estas implementaciones sean idénticas bloque por bloque.

Confirmado en el notebook: cuatro canales, patches `128 x 128`, cinco resoluciones, base de 64 canales, doble convolución con BatchNorm y ReLU, interpolación bilineal, U-Net++ sin deep supervision y salida lineal de un canal.

Pendiente de confirmar en el paper:

- orden y significado de los cuatro canales;
- tamaño exacto de patch;
- profundidad y filtros por nivel;
- uso de BatchNorm;
- upsampling bilineal o convolución transpuesta;
- padding y bordes;
- deep supervision en U-Net++;
- activación de salida;
- loss e inicialización;
- normalización de entrada y objetivo;
- split, nodata y agregación de MAE, RMSE y R².

Hasta completar estos puntos, son baselines razonables del mismo tipo arquitectónico, pero no reproducciones exactas demostradas del paper.

## 10. Limitaciones técnicas observadas

1. El smoke test genérico interpola antes de comprobar la forma. Una prueba contractual estricta debería comprobar primero la salida nativa; el análisis de ambos `forward` indica que ya es `128 x 128`.
2. El entrenamiento también puede ajustar silenciosamente la predicción al target. No afecta a estas implementaciones, pero podría ocultar errores futuros.
3. `DoubleConv` se redefine. La versión usada por U-Net es funcionalmente equivalente a la anterior, aunque la duplicación aumenta el riesgo de divergencia.
4. No hay evidencia versionada de un smoke test real con PyTorch; debe ejecutarse en Colab antes de resultados finales.

Ningún punto exige modificar ahora las arquitecturas.

## 11. Preguntas pendientes para la metodología del paper

1. ¿Cuáles son los cuatro canales y su orden?
2. ¿El paper usa patches de `128 x 128`?
3. ¿Qué profundidad y filtros usa en U-Net y U-Net++?
4. ¿Incluye BatchNorm tras cada convolución?
5. ¿Usa interpolación bilineal, convolución transpuesta u otro upsampling?
6. ¿La U-Net++ del paper usa deep supervision?
7. ¿Qué salida y activación emplea para el DSM?
8. ¿Qué loss e inicialización usa?
9. ¿Cómo normaliza entradas y DSM?
10. ¿Cómo define los splits y evita fuga espacial?
11. ¿Cómo trata nodata y qué unidad vertical utiliza?
12. ¿Agrega métricas globalmente por píxel o por imagen?

## 12. Cómo defender estos baselines ante el tutor

> Se han elegido U-Net y U-Net++ porque son las baselines del paper que permiten comparar un encoder-decoder con skips directos frente a una variante con skips densos y anidados. Ambas se adaptan a regresión DSM: reciben cuatro canales y devuelven un mapa continuo de un canal, sin sigmoid ni softmax. U-Net++ usa la salida final x04 y no incorpora deep supervision. Las formas y conexiones están verificadas en el código, pero la equivalencia exacta con el paper queda pendiente hasta confirmar hiperparámetros, preprocessing, split y nodata.

Conviene separar tres niveles de afirmación:

1. **Verificado en el código:** estructura, canales, resoluciones, salida lineal y ausencia de deep supervision.
2. **Validado experimentalmente:** shapes y ejecución en Colab, cuando se conserve evidencia del smoke test.
3. **Pendiente frente al paper:** detalles no disponibles en el repositorio.

Así no se presenta como reproducción exacta lo que actualmente es una implementación baseline razonable y trazable.

## 13. Conclusión

U-Net y U-Net++ están bien conectadas para `(B, 4, 128, 128) -> (B, 1, 128, 128)` y son adecuadas para regresión DSM. U-Net usa cuatro skips directos; U-Net++ construye la malla completa hasta `x04` y no usa deep supervision. No se recomienda cambiar ninguna arquitectura hasta confirmar la configuración exacta del paper.
