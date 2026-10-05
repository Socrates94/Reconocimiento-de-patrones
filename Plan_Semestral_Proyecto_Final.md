# 🗺️ Plan de Trabajo Semestral: Proyecto Final de Reconocimiento de Patrones

**Basado en el artículo:** *"Optimized hybrid SVM-RF multi-biometric framework for enhanced authentication using fingerprint, iris, and face recognition"* (Sonal et al., 2025).

Este documento detalla la hoja de ruta para evolucionar desde el pipeline tabular actual (Tarea 01) hacia un sistema biométrico multimodal de última generación, dividiendo el trabajo en hitos lógicos para el resto del semestre.

---

## 🎯 El Cambio de Paradigma (De Tarea 01 al Proyecto Final)

En la Tarea 01 trabajamos con **datos tabulares puros** (CSV con números). En este artículo, el enfoque cambia radicalmente a **Procesamiento Digital de Imágenes (Visión Artificial)**. 
El profesor quiere que apliquemos los mismos principios (Cargar, Normalizar, Reducir) pero con un enfoque en imágenes biométricas. 

Aquí está la equivalencia directa de lo que hicimos vs lo que haremos:
*   **Dataset:** De `.txt` numéricos ➡️ Colecciones de imágenes (rostros, iris, huellas).
*   **Normalización:** De Z-Score/Min-Max ➡️ Redimensionamiento y Filtros de mejora de contraste (CLAHE).
*   **Extracción/Correlación:** De PCA/Pearson ➡️ Filtros de Gabor y Algoritmos Genéticos.

---

## 📅 Fase 1: Preprocesamiento de Imágenes (Objetivo Inmediato)
*Esto cubre lo que el profesor está pidiendo justo ahora.*

### 1.1 Adquisición de Datasets
Necesitamos descargar los 3 datasets de imágenes que menciona el artículo (página 21):
*   **Huellas dactilares:** SOCOFing (Kaggle).
*   **Iris:** MMU Iris Dataset (Kaggle).
*   **Rostros:** Real and Fake Face Detection (GitHub).

### 1.2 "Normalización" (Preprocesamiento Visual)
En lugar de matemáticas como el Z-Score, las imágenes requieren normalización visual. Crearemos un módulo en Python (usando `OpenCV`) que aplique el siguiente pipeline a cada imagen:
1.  **Redimensionamiento (Image Resizing):** Escalar todas las imágenes a un tamaño estandarizado de `128 x 128` píxeles usando interpolación bilineal.
2.  **Eliminación de Ruido (Noise Removal):** Aplicar un Filtro de Mediana con un kernel de `3x3` para limpiar el ruido de "sal y pimienta".
3.  **Mejora de Contraste (Contrast Enhancement):** Aplicar Ecualización de Histograma Adaptativo (CLAHE) con un límite de clip de `2.0` para que las texturas ocultas del iris y huellas resalten.

---

## 📅 Fase 2: Extracción y Reducción de Características (Análisis de Correlación)
*Aquí traducimos las imágenes a números para que la computadora las entienda.*

### 2.1 Extracción con Filtros de Gabor
Las imágenes no se pueden meter crudas a una IA. Usaremos **Filtros de Gabor** para escanear las imágenes y extraer "vectores de características" basados en la textura (crestas de la huella, patrones del iris). Esto generará matrices gigantescas de datos.

### 2.2 Optimización y Reducción (El equivalente a PCA/Pearson)
Dado que los Filtros de Gabor generan vectores inmensos, sufriremos la maldición de la dimensionalidad. El artículo propone usar algoritmos avanzados para seleccionar solo las mejores características:
1.  **BFO (Bacterial Foraging Optimization):** Para buscar óptimos locales (qué características funcionan bien juntas).
2.  **GA (Algoritmos Genéticos):** Para selección natural de características a nivel global.
*Nota estratégica:* Podemos empezar implementando nuestro ya conocido **PCA** como línea base, y luego subir la dificultad a **GA** para el final del semestre.

---

## 📅 Fase 3: Entrenamiento de Modelos Base (Machine Learning)
Una vez que tengamos la matriz de características "limpia y optimizada" (equivalente a nuestro dataset limpio actual), la dividiremos en **70% Entrenamiento y 30% Prueba** (como dicta el artículo) para alimentar los algoritmos:

1.  **Máquinas de Vectores de Soporte (SVM):**
    *   Implementación de SVM con **Kernel Lineal** (porque el artículo prioriza eficiencia computacional).
    *   Objetivo: Encontrar hiperplanos que separen a las personas.
2.  **Bosques Aleatorios (Random Forest - RF):**
    *   Uso de múltiples árboles de decisión basados en la métrica de *Impureza de Gini*.
    *   Objetivo: Manejar el ruido y prevenir el sobreajuste (*overfitting*).

---

## 📅 Fase 4: Arquitectura Híbrida (La joya del proyecto)
El aporte principal de tu artículo no es usar SVM o RF por separado, sino crear el modelo **Híbrido SVM-RF**.
*   **La Lógica:** Primero, los datos pasan por el SVM porque es muy rápido y preciso para marcar los límites iniciales. Luego, esos resultados (sobre todo los dudosos) se pasan al Random Forest para que el ensamble de árboles haga el refinamiento final de la predicción.
*   **Fusión a Nivel de Característica:** El sistema debe combinar las predicciones de Huella + Rostro + Iris en una sola decisión de autenticación.

---

## 📅 Fase 5: Evaluación y Reporte Científico Final
En el último mes del semestre, el código debe someterse a pruebas de estrés para generar las gráficas del paper original:
*   Generar matrices de confusión.
*   Calcular métricas estrictas: **Exactitud (Accuracy), Precisión, Recall, F1-Score, y Especificidad.**
*   Graficar la **Curva ROC** y el área bajo la curva (AUC).
*   **Meta Final:** Demostrar empíricamente que el modelo híbrido (esperando un ~97.56% de accuracy) supera a usar SVM (54%) o RF (66%) por separado.

---
**Siguiente Paso Inmediato para la clase:** Crear un nuevo archivo de Python `01_image_preprocessing.py` utilizando `cv2` (OpenCV) para cargar los 3 datasets y aplicarles el Filtro de Mediana, el Redimensionamiento 128x128 y la ecualización CLAHE.
