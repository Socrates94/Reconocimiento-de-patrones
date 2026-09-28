# Pipeline de Reconocimiento de Patrones - Tarea 01

Este proyecto es un pipeline completo de Machine Learning enfocado en el preprocesamiento, normalización, análisis de correlación (PCA/CCA) y preparación de datos para algoritmos de **Reconocimiento de Patrones**.

## 📂 Estructura del Proyecto

El proyecto está organizado en una arquitectura por carpetas para mantener separados el código fuente y los datos en sus diferentes etapas de transformación:

```text
Reconocimiento-de-patrones/
├── data/
│   ├── 1_raw/             # Datasets originales en formato .txt (ej. Slice.txt, Vh.txt)
│   ├── 3_transformed/     # Datasets convertidos a CSV y Normalizados
│   ├── 4_correlacion/     # Matrices de Pearson y datasets reducidos (PCA/CCA)
│   └── 5_splits/          # Divisiones finales en Train (70%), Validation (15%), Test (15%)
├── src/
│   ├── 01_procesamiento.py          # Conversión de TXT a CSV
│   ├── 02_normalizacion.py          # Normalización (Min-Max, Z-Score, Robust)
│   ├── 03_analisisCorrelacion.py    # Correlación de Pearson, PCA y CCA
│   ├── 04_discretizacion.py         # Discretización de variables
│   ├── 05_splitData.py              # División de conjuntos de datos
│   └── normalizacion_manual.py      # Implementación matemática manual de normalizadores
├── venv/                  # Entorno virtual de Python
├── main.py                # Archivo principal que orquesta todo el pipeline
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Este archivo
```

---

## 🚀 Etapas del Pipeline y Explicación de Métodos

El pipeline se ejecuta secuencialmente a través del archivo `main.py` y consta de las siguientes etapas:

### 1. Procesamiento (`01_procesamiento.py`)
Lee los archivos `.txt` crudos que contienen los datos y los convierte a un formato tabular estándar `.csv` para facilitar su lectura en Pandas.

### 2. Normalización (`02_normalizacion.py` y `normalizacion_manual.py`)
Escala los datos para evitar que características con números grandes dominen el cálculo de distancias matemáticas. Se aplican y comparan varios métodos:

*   **Min-Max:** 
    *   **Fórmula:** `X_nuevo = (X - X_min) / (X_max - X_min)`
    *   **¿Cuándo usarlo?** Es ideal cuando se conocen los límites fijos (ej. píxeles de 0 a 255) o cuando los algoritmos requieren que los datos estén estrictamente en un rango como `[0, 1]` (ej. Redes Neuronales).
    *   **Manejo de Outliers:** Es PÉSIMO lidiando con outliers. Un solo dato gigante comprime todos los demás valores "normales" a un rango casi de 0.

*   **Z-Score (StandardScaler):** 
    *   **Fórmula:** `X_nuevo = (X - Media) / Desviacion_Estandar`
    *   **¿Cuándo usarlo?** Es la opción "por defecto" para algoritmos como PCA o SVM. Funciona excelente si los datos se agrupan al centro (siguen una distribución normal o Campana de Gauss).
    *   **Manejo de Outliers:** Le afectan moderadamente. Un outlier moverá la media general y ampliará la desviación estándar, pero es más seguro que Min-Max.

*   **RobustScaler:** 
    *   **Fórmula:** `X_nuevo = (X - Mediana) / Rango_Intercuartilico`
    *   **¿Cuándo usarlo?** ¡Es EL MEJOR MÉTODO cuando sabes que tu dataset está lleno de DATOS ATÍPICOS (outliers)!
    *   **Manejo de Outliers:** Es completamente inmune (robusto) a ellos, ya que usa la Mediana y los Cuartiles centrales que ignoran por completo los valores extremos.

*   **Decimal Scaling (Manual):** 
    *   **Fórmula:** `X_nuevo = X / (10^j)`
    *   **¿Cuándo usarlo?** Cuando solo quieres hacer los números más pequeños moviendo el punto decimal, preservando exactamente los mismos dígitos de la estructura original.
    *   **Manejo de Outliers:** Muy sensible a los outliers, ya que el valor absoluto más grande del dataset es el que dicta cuántas posiciones se mueve el punto decimal para todos los datos.

### 3. Análisis Exploratorio y de Correlación (`03_analisisCorrelacion.py`)
Analiza la relación entre las características y reduce la dimensionalidad para combatir la "Maldición de la Dimensionalidad":

*   **Correlación de Pearson:** 
    *   Mide la relación lineal entre dos columnas (de -1 a 1).
    *   **¿Cuándo usarlo?** Ideal como paso de limpieza para detectar "clones" (características con `r > 0.98`) y borrar uno de ellos, ahorrando procesamiento. Funciona cuando los datos son numéricos y la relación se asume como lineal.
    *   **¿Cuándo NO usarlo?** Si tus datos son categóricos (texto) o si la relación entre variables es compleja y no-lineal (ej. curva parabólica).

*   **PCA (Análisis de Componentes Principales):** 
    *   Comprime cientos de columnas en unos pocos "componentes" rescatando la mayor cantidad de varianza posible. Es una técnica **No Supervisada**.
    *   **¿Cuándo usarlo?** Cuando sufres por tener demasiadas columnas y el modelo es muy lento. *Regla de oro: los datos deben estar previamente normalizados.*
    *   **¿Cuándo NO usarlo?** Como PCA es "ciego" a las etiquetas a predecir, al comprimir la varianza podría borrar justo la información útil que diferenciaba a tus clases. Tampoco sirve si necesitas explicarle a alguien qué variable original exacta causó una predicción (PCA mezcla todo).

*   **CCA (Análisis de Correlación Canónica):** 
    *   Transforma las características (X) en componentes que tengan la MÁXIMA correlación con la clase a predecir (Y). Es una técnica **Supervisada**.
    *   **¿Cuándo usarlo?** Cuando tu prioridad absoluta es que los nuevos componentes mantengan una relación fortísima con las etiquetas de tus clases (muy útil para clasificación).
    *   **¿Cuándo NO usarlo?** Si tus características (X) tienen multicolinealidad extrema (columnas clonadas redundantes), las matemáticas de CCA "explotan". Por eso es obligatorio aplicar Pearson primero para limpiar clones antes de usar CCA.

### 4. Discretización (`04_discretizacion.py`)
*(Opcional)* Convierte variables continuas en intervalos discretos o categorías.

### 5. División de Datos / Split (`05_splitData.py`)
Divide rigurosamente los datasets procesados para Machine Learning:
*   **Train (70%):** Para enseñar al modelo.
*   **Validation (15%):** Para ajustar hiperparámetros.
*   **Test (15%):** Para evaluar el rendimiento final sin sesgos.

---

## ⚙️ Instalación y Ejecución

Para correr este proyecto en tu máquina local:

### 1. Crear y activar el entorno virtual
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno (Linux/Mac)
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el Pipeline
Coloca tus datasets originales (`Slice.txt`, `Vh.txt`) dentro de la carpeta `data/1_raw/` y luego ejecuta el orquestador:
```bash
python main.py
```
*Nota: Actualmente el pipeline está configurado específicamente para leer Slice y Vh. Nuevos datasets requieren actualizar los nombres hardcodeados en el código fuente.*
