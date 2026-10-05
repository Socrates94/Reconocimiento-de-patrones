# 🔍 Reporte a Detalle: Paso 1 - Preprocesamiento de Imágenes

Este reporte detalla matemáticamente y en código cómo abordaremos el **Paso 1** de tu nuevo Proyecto Final, según la metodología del artículo científico. En esta fase, nuestro objetivo es convertir las fotos crudas de huellas, iris y rostros en matrices matemáticas limpias, estandarizadas y libres de ruido.

---

## 📸 1. Estructura de Entrada (Tus Datasets)
Ya que descargaste los datasets, el primer paso físico que debes hacer en tu computadora es organizarlos dentro de la nueva carpeta `data/1_raw_images/`. La estructura ideal sería:
```text
data/1_raw_images/
├── fingerprint/    # Imágenes del dataset SOCOFing
├── face/           # Imágenes del dataset Real and Fake Face Detection
└── iris/           # Imágenes del dataset MMU Iris
```

---

## 🛠️ 2. El Pipeline de Preprocesamiento (Lo que programaremos)
En el módulo vacío que acabo de crear (`src/01_image_preprocessing.py`), implementaremos una función iterativa usando `OpenCV (cv2)`. Para cada imagen que encuentre en tus carpetas, aplicará estos 3 procesos matemáticos:

### A. Eliminación de Ruido (Noise Removal)
El artículo especifica el uso de un **Filtro de Mediana (Median Filter)** con un *kernel* (ventana) de `3x3`.
*   **El problema:** Las cámaras o escáneres biométricos suelen introducir ruido de "sal y pimienta" (píxeles completamente blancos o negros aleatorios).
*   **La solución matemática:** El filtro de mediana se para sobre cada píxel, mira a sus 8 vecinos inmediatos (un cuadro de 3x3), ordena los 9 valores de menor a mayor y escoge el número de en medio (la mediana) para reemplazar el píxel central. Esto destruye el ruido extremo sin difuminar los bordes finos de las huellas dactilares.
*   **Implementación proyectada en Python:** `cv2.medianBlur(imagen, 3)`

### B. Redimensionamiento Uniforme (Image Resizing)
El artículo exige que TODAS las imágenes (sin importar si son rostros en HD o huellas pequeñas) se fuercen a un tamaño de **128 x 128 píxeles**.
*   **El problema:** Para que el Machine Learning funcione, cada vector de características debe tener exactamente la misma longitud.
*   **La solución matemática:** Usaremos **Interpolación Bilineal**. Si una imagen debe agrandarse o achicarse, esta técnica calcula los nuevos píxeles realizando un promedio ponderado de los 4 píxeles originales más cercanos.
*   **Implementación proyectada en Python:** `cv2.resize(imagen, (128, 128), interpolation=cv2.INTER_LINEAR)`

### C. Mejora de Contraste (Contrast Enhancement)
El artículo dicta el uso de **CLAHE** (Contrast Limited Adaptive Histogram Equalization) con un "Clip Limit" de `2.0`.
*   **El problema:** Las huellas o el iris pueden tener zonas muy oscuras o iluminadas de forma desigual. Si usamos una ecualización normal, el "brillo" se revienta y perdemos las texturas.
*   **La solución matemática:** CLAHE divide la imagen en cuadritos pequeños (ej. 8x8). Ecualiza el histograma (el contraste) *solo dentro de cada cuadrito*. El parámetro `Clip Limit = 2.0` significa que si en un cuadrito hay demasiado contraste concentrado, se recorta ese exceso y se redistribuye para que la imagen se vea súper nítida y natural, resaltando las venas del iris y crestas dactilares.
*   **Implementación proyectada en Python:** 
  ```python
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  imagen_mejorada = clahe.apply(imagen)
  ```

---

## 💾 3. Salida del Paso 1
Una vez que el script aplique esos tres algoritmos, tomará las nuevas imágenes (ahora cuadradas de 128x128, sin ruido y súper contrastadas) y las guardará en `data/2_preprocessed/`.

Esa carpeta será nuestra nueva "materia prima" pura. ¡A partir de ahí, ya ni siquiera las veremos como imágenes, sino como matrices de números perfectas listas para que les apliquemos los Filtros de Gabor en el Paso 2!

**Tu tarea hoy:** Mueve o descomprime las carpetas de las imágenes que descargaste hacia `data/1_raw_images/`. ¡Cuando estés listo, escríbeme y empezamos a codificar el módulo 01!
