# 🧬 Proyecto Final: Framework Híbrido SVM-RF para Autenticación Multi-Biométrica

Este proyecto implementa un sistema avanzado de **Reconocimiento de Patrones y Visión Artificial**, basado en el artículo científico *"Optimized hybrid SVM-RF multi-biometric framework for enhanced authentication using fingerprint, iris, and face recognition"* (Sonal et al., 2025).

El objetivo es construir un pipeline de Machine Learning que procese imágenes biométricas (huellas, rostros e iris), extraiga características mediante Filtros de Gabor, reduzca la dimensionalidad con algoritmos metaheurísticos, y finalmente clasifique utilizando un ensamble híbrido de Máquinas de Vectores de Soporte (SVM) y Bosques Aleatorios (Random Forest).

---

## 📂 Estructura Arquitectónica del Proyecto

El proyecto mantiene un diseño modular, separando estrictamente los datos crudos, las transformaciones intermedias y el código fuente. Esta es la estructura objetivo hacia la que evolucionará el código:

```text
Reconocimiento-de-patrones/
├── Articulo RP/           # Código fuente LaTeX y PDF del artículo científico final
├── Bibliografia/          # Libros y referencias clave de la materia
├── data/
│   ├── 1_raw_images/      # Datasets originales (Huellas SOCOFing, Rostros, Iris MMU)
│   ├── 2_preprocessed/    # Imágenes con filtro de mediana, CLAHE y tamaño 128x128
│   ├── 3_features/        # Vectores extraídos crudos por los Filtros de Gabor
│   ├── 4_optimized/       # Características reducidas por Algoritmos Genéticos (GA) y BFO
│   └── 5_splits/          # División final en Entrenamiento (70%) y Prueba (30%)
├── src/
│   ├── 01_image_preprocessing.py   # Limpieza, redimensionamiento y ecualización de imágenes
│   ├── 02_feature_extraction.py    # Aplicación de bancos de Filtros de Gabor
│   ├── 03_optimization.py          # Selección de características (BFO y GA)
│   ├── 04_model_training.py        # Entrenamiento de SVM, RF y el modelo Híbrido
│   └── 05_evaluation.py            # Matrices de confusión, Curvas ROC y Métricas
├── venv/                  # Entorno virtual de Python gestionado por 'uv'
├── main.py                # Orquestador maestro que ejecutará el pipeline completo
├── requirements.txt       # Dependencias del proyecto (OpenCV, scikit-learn, etc.)
├── Plan_Semestral_Proyecto_Final.md # Hoja de ruta detallada del semestre
├── Stack_Tecnologico_Proyecto_Final.md # Detalle de librerías Python a utilizar
└── README.md              # Este archivo
```

---

## 📊 Datasets Biomédicos (Fuentes de Datos)

Este proyecto utiliza tres fuentes de datos biométricos de acceso público para entrenar y evaluar el modelo híbrido. Todas las imágenes deben ser almacenadas en el directorio `data/1_raw_images/`.

### 1. Detección de Rostros (Real and Fake Face Detection)
*   **Fuente Oficial:** [Kaggle - Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
*   **Características:** Consiste en una colección de imágenes faciales divididas en rostros reales (fotografías auténticas de personas) y rostros falsificados (creados digitalmente o usando técnicas tipo Deepfake/Photoshop). Es ideal para probar la robustez del sistema contra ataques de suplantación.
*   **Estructura Interna:** Usualmente el archivo descargado se divide en carpetas que separan a las clases reales y falsas (ej. `training_real` y `training_fake`).

### 2. Huellas Dactilares (SOCOFing)
*   **Fuente Oficial:** [Kaggle - SOCOFing Dataset](https://www.kaggle.com/datasets/ruizgara/socofing?resource=download)
*   **Características:** Sokoto Coventry Fingerprint Dataset (SOCOFing) es una base de datos de investigación que contiene miles de imágenes de huellas dactilares. Su característica más importante es que proporciona huellas reales de alta calidad y luego proporciona versiones "alteradas" o "corruptas" sintéticamente para simular cortes, borrones o distorsiones.
*   **Estructura Interna:** Encontrarás subcarpetas maestras como `Real` (huellas perfectas) y `Altered` (las huellas con ruido artificial).

### 3. Patrones de Iris (MMU Iris Database)
*   **Fuente Oficial:** [Kaggle - MMU Iris Dataset](https://www.kaggle.com/datasets/naureenmohammad/mmu-iris-dataset?utm_source=chatgpt.com)
*   **Características:** Una colección biométrica capturada mediante cámaras de infrarrojo cercano para iluminar perfectamente el patrón textural del iris de 46 personas distintas. Las imágenes suelen presentar variaciones por la pupila dilatada, anteojos y pestañas.
*   **Estructura Interna:** Al descomprimirlo, verás docenas de carpetas numeradas (ej. `1`, `2`, ..., `46`). ¡Ojo! Cada número es una persona (o clase) diferente. Y dentro de la carpeta `1`, normalmente encontrarás subcarpetas como `left` (ojo izquierdo) y `right` (ojo derecho).

---

## 🚀 Etapas del Pipeline Biomédico (Roadmap)

A continuación se detalla el flujo de trabajo que implementaremos paso a paso en los módulos de Python:

### 1. Preprocesamiento Visual (`01_image_preprocessing.py`)
Equivalente a la "Normalización" de datos tabulares, pero aplicado a imágenes:
*   **Filtro de Mediana (3x3):** Para eliminar ruido del sensor (Salt-and-Pepper).
*   **Redimensionamiento:** Interpolación bilineal para dejar todas las imágenes en una matriz estándar de `128x128`.
*   **CLAHE (Contrast Enhancement):** Ecualización de histograma adaptativa para resaltar crestas de huellas y texturas del iris sin quemar la imagen.

### 2. Extracción de Características (`02_feature_extraction.py`)
Las imágenes no se pueden clasificar en bruto. Usaremos **Filtros de Gabor** para analizar el dominio espacial y de frecuencias, capturando las texturas matemáticas únicas de cada rasgo biométrico.

### 3. Optimización de Dimensiones (`03_optimization.py`)
Para evitar la maldición de la dimensionalidad provocada por los vectores masivos de Gabor, usaremos selección natural:
*   **GA (Genetic Algorithms):** Selección global de características basada en evolución.
*   **BFO (Bacterial Foraging Optimization):** Búsqueda local de las mejores combinaciones de características.

### 4. Clasificación Híbrida (`04_model_training.py`)
*   Se divide el dataset limpio y optimizado en **70% Train** y **30% Test**.
*   Se construye un modelo conjunto (Ensemble) que toma la velocidad y precisión para trazar fronteras lineales del **SVM** y la resistencia contra el ruido del **Random Forest (RF)**.
*   > 💡 **Nota de Rendimiento (Hardware):** Para aprovechar al máximo arquitecturas multi-núcleo (como procesadores Ryzen 9 de 24 núcleos), forzaremos el multiprocesamiento en Scikit-Learn. A los clasificadores que lo soporten se les añadirá el parámetro `n_jobs=-1` para utilizar el 100% del procesador.
    Ejemplo:
    ```python
    bosque = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    ```

### 5. Evaluación Científica (`05_evaluation.py`)
Validación estricta de resultados para el reporte final de la materia, calculando:
*   Accuracy (Exactitud esperada ~97%)
*   Precision, Recall, F1-Score y Especificidad.
*   Gráficas de la Curva ROC (AUC).

---

## ⚙️ Configuración y Ejecución del Proyecto

El framework está diseñado para ejecutarse secuencialmente, respetando el ciclo de vida de los datos (desde las imágenes en crudo hasta la evaluación final).

### 1. Preparación del Entorno
Utiliza `uv` (el gestor ultrarrápido de paquetes en Rust) para aislar las dependencias de forma segura:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Ejecución del Pipeline (Orquestador Maestro)
El archivo principal es `main.py`. Este funciona como tu tablero de control o "Director de Orquesta". Para ejecutar una etapa de la investigación, simplemente abre `main.py`, elimina el símbolo `#` de la etapa que deseas correr, vuelve a poner el `#` en las demás, y ejecuta:

```bash
python main.py
```

**Flujo lógico de ejecución que debe seguirse (Una sola vez por etapa):**
1. **Etapa 1:** Lee de `data/1_raw_images/` y escribe en `data/2_preprocessed/` (Limpieza visual).
2. **Etapa 2:** Lee de `data/2_preprocessed/` y guarda vectores matemáticos puros en `data/3_features/` (Gabor).
3. **Etapa 3:** Lee los vectores pesados, los optimiza genéticamente reduciendo su tamaño y los guarda en `data/4_optimized/`.
4. **Etapa 4:** Carga los datos finales, construye el clasificador SVM-RF y empaqueta la Inteligencia Artificial entrenada en disco duro dentro de `data/5_splits/`.
5. **Etapa 5:** Somete a los modelos empaquetados al "Examen Final" (con el 30% ciego de Test), imprimiendo en la consola los porcentajes de Accuracy y generando los **Gráficos Científicos** en la carpeta `results/`.

---

## 🔮 Visión a Futuro: Descentralización Biométrica (Tesis de Maestría)

Como propuesta de innovación y puente directo hacia la Tesis de Maestría, la arquitectura de este framework biométrico se proyectará en el artículo científico (sección *Future Work*) hacia la **Identidad Auto-Soberana (SSI)**.

Aunque el modelo SVM-RF garantiza alta precisión, las bases de datos biométricas centralizadas representan un riesgo de seguridad crítico (*Honeypots*). Por ende, el trabajo a futuro establecerá que:
1. El **embedding matemático** (los vectores de características extraídos y optimizados por esta IA) NO se almacene en un servidor.
2. Sea emitido como una **Credencial Verificable** directamente a la billetera móvil del usuario mediante una DLT (**Hyperledger Indy**).
3. La validación matemática de identidad se resuelva en el borde sin revelar el biométrico real, mediante **Pruebas de Conocimiento Cero (ZKP)**.
