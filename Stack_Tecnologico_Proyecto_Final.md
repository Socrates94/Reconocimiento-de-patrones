# 🛠️ Stack Tecnológico: Proyecto Final (Sistema Híbrido Biomédico)

¡Efectivamente! El proyecto es pesado, pero la buena noticia es que el ecosistema de **Python** está perfectamente preparado para todo esto. Aunque ya no usaremos el código que leía archivos `.txt`, **sí vamos a reciclar toda la lógica y la arquitectura modular** (carpetas separadas, el pipeline que ejecuta paso a paso y la lógica de dividir datos).

Como bien te comentó Gemini Web, `scikit-learn` será nuestro mejor amigo, pero para manipular imágenes necesitaremos herramientas adicionales. Aquí tienes el stack exacto que usaremos para cada paso del *paper*:

---

## 1. Entorno y Gestión de Paquetes
*   **`uv`**: Ya que decidiste migrar a esta tecnología, la seguiremos usando para gestionar nuestro entorno virtual de manera ultra rápida. Nos asegurará que nuestras pesadas librerías de visión por computadora se instalen en segundos.

## 2. Preprocesamiento de Imágenes (La nueva "Normalización")
Para leer las huellas, iris y rostros, y limpiarlas:
*   **`OpenCV-Python (cv2)`**: Es el estándar de la industria para Visión Artificial. Lo usaremos para:
    *   Cargar las imágenes desde tus carpetas.
    *   Redimensionar todas a `128x128` píxeles (`cv2.resize`).
    *   Aplicar el filtro de mediana para quitar el ruido (`cv2.medianBlur`).
    *   Aplicar la mejora de contraste CLAHE (`cv2.createCLAHE`).
*   **`NumPy`**: Debajo del capó, cada imagen que carga OpenCV no es más que una matriz gigante de números. NumPy nos servirá para manipular estas matrices rápidamente.

## 3. Extracción de Características
*   **`Scikit-Image (skimage.filters)` u `OpenCV`**: Ambos tienen implementaciones nativas de los **Filtros de Gabor**. Generaremos un "banco de filtros" (diferentes orientaciones y frecuencias) y pasaremos cada imagen por ellos para extraer las texturas (las arrugas de la huella, el patrón del iris).

## 4. Optimización de Características (Reducción)
Dado que Gabor va a generar *miles* de características, necesitamos los algoritmos de optimización (GA y BFO) que pide el artículo para no quemar la RAM de tu laptop:
*   **`PyGAD` o `DEAP`**: Librerías especializadas en Python para implementar **Algoritmos Genéticos (GA)**. Nos ayudarán a simular la evolución darwiniana para seleccionar solo las mejores características.
*   **`Mealpy`**: Es una biblioteca moderna increíble que contiene la implementación de decenas de algoritmos meta-heurísticos modernos, incluyendo **Bacterial Foraging Optimization (BFO)**. Si no, podemos programarlo a mano con NumPy, ya que las fórmulas vienen en el artículo.

## 5. Machine Learning (Los Clasificadores)
Aquí entra la magia de **Scikit-Learn (`sklearn`)**:
*   **Support Vector Machine (SVM):** Usaremos `sklearn.svm.SVC(kernel='linear')`.
*   **Random Forest (RF):** Usaremos `sklearn.ensemble.RandomForestClassifier()`.
*   **El Modelo Híbrido:** Scikit-Learn permite construir "Ensembles" personalizados. Escribiremos un script donde la salida (las probabilidades) del SVM sirvan como alimentación inicial para que el Random Forest tome la decisión final.
*   **División de Datos:** ¡Esto sí lo reciclamos! Seguiremos usando `train_test_split` con `stratify=y` de Scikit-Learn como lo hicimos en la Tarea 01.

## 6. Evaluación y Gráficas Científicas
Para replicar las gráficas del artículo (como las páginas 11, 14, 15 y 19):
*   **`Scikit-Learn (sklearn.metrics)`**: Para calcular internamente Accuracy, Precision, Recall, F1-Score y la Curva ROC.
*   **`Matplotlib` y `Seaborn`**: Para dibujar las gráficas, las barras comparativas y la Matriz de Confusión para que queden listas para exportar a tu futuro reporte en LaTeX.

---

### Resumen de dependencias a instalar mañana:
Cuando comencemos, el comando de `uv` se verá más o menos así:
```bash
uv pip install opencv-python numpy scikit-learn scikit-image matplotlib seaborn pygad mealpy
```

¡Descansa! Mañana que tengas los datasets empezamos a configurar OpenCV para cargar esas imágenes y ver qué tal se ven.
