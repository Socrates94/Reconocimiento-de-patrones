# Pipeline de Reconocimiento de Patrones - Tarea 01

Este proyecto contiene un pipeline completo de Machine Learning clásico para preprocesamiento, extracción de características y transformación de datos, específicamente desarrollado para analizar los datasets `Slice` y `Vh`.

## 📁 Arquitectura del Proyecto

El proyecto sigue las mejores prácticas de la industria, separando estrictamente los datos originales del código fuente:

```text
Tarea 01/
├── data/
│   ├── 1_raw/                 # Archivos originales en texto plano (.txt)
│   ├── 2_processed/           # Archivos convertidos a formato tabular (.csv)
│   └── 3_transformed/         # Archivos normalizados y discretizados
├── src/
│   ├── 01_procesamiento.py    # Conversión de formatos
│   ├── 02_normalizacion.py    # Escalado matemático (MinMax, Z-score, Robust)
│   ├── 03_analisis.py         # Extracción de características (PCA, CCA, Pearson)
│   └── 04_discretizacion.py   # Transformación a categorías (Histogram Binning)
├── main.py                    # Script principal (Director de orquesta)
└── README.md                  # Documentación
```

## 🚀 Cómo ejecutar el proyecto

Para correr el pipeline completo de principio a fin, simplemente ejecuta el archivo principal desde la terminal:

```bash
python main.py
```

El script `main.py` se encargará de llamar a cada uno de los módulos de la carpeta `src/` en el orden correcto. Tomará los datos de `data/1_raw/` y depositará los resultados finales listos para entrenar algoritmos de clasificación en `data/3_transformed/`.

## 🧠 Fases del Pipeline

1.  **Preprocesamiento:** Convierte los archivos `.txt` (donde las columnas están separadas por comas) a archivos `.csv` estandarizados.
2.  **Normalización:** Aplica tres métodos de escalado con `scikit-learn` para poner todas las características en el mismo rango de valores y evitar sesgos:
    *   *Min-Max Scaler*
    *   *Z-Score (Standard Scaler)*
    *   *Robust Scaler*
3.  **Discretización:** Convierte variables continuas en categorías (contenedores) usando *Histogram Binning*. Muy útil para algoritmos como Árboles de Decisión clásicos o Naive Bayes.
4.  **Análisis Multivariante:**
    *   **Correlación de Pearson:** Detecta características redundantes (clones matemáticos).
    *   **PCA (Principal Component Analysis):** Reduce la dimensionalidad del dataset (aplastando cientos de columnas a solo 10) reteniendo la mayor cantidad de varianza.
    *   **CCA (Canonical Correlation Analysis):** Evalúa la correlación máxima entre el espacio de características y las etiquetas de clase.
