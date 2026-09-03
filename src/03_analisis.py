import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_TRANSFORMED = os.path.join(BASE_DIR, 'data', '3_transformed')

def matriz_covarianza_pearson(df, threshold=0.98):
    print("\n--- 1. Matriz de Covarianza y Pearson ---")
    
    # 1. Separamos las características (X) de la clase final para no correlacionar con la etiqueta
    X = df.iloc[:, :-1]
    
    # 2. Calculamos la matriz de correlación usando la fórmula de Pearson
    correlacion = X.corr(method='pearson')
    
    # 3. Inicializamos un contador para detectar características "clonadas" (muy similares)
    pares_altos = 0
    
    # 4. Recorremos solo la mitad de la matriz (diagonal superior) para no contar pares dobles (ej. A-B y B-A)
    for i in range(len(correlacion.columns)):
        for j in range(i + 1, len(correlacion.columns)):
            # 5. Extraemos el coeficiente de correlación 'r' entre la característica i y la característica j
            r = correlacion.iloc[i, j]
            
            # 6. Si el valor absoluto (sin importar signo) supera nuestro límite, lo contamos
            if abs(r) > threshold:
                pares_altos += 1
                
    # 7. Imprimimos el número total de pares redundantes encontrados
    print(f"  Se encontraron {pares_altos} pares altamente correlacionados (|r| > {threshold}).")

# def reduccion_pca(X, n_componentes=5):
#     print(f"\n--- 2. Reducción de Dimensionalidad con PCA ({n_componentes} componentes) ---")
#     pca = PCA(n_components=n_componentes)
#     X_reducido = pca.fit_transform(X)
    
#     varianza_explicada = pca.explained_variance_ratio_
#     varianza_total = np.sum(varianza_explicada) * 100
        
#     print(f"-> Varianza total conservada por las {n_componentes} componentes: {varianza_total:.2f}%")
#     return X_reducido

# def analisis_cca(X, y):
#     print("\n--- 3. Análisis de Correlación Canónica (CCA) ---")
#     encoder = OneHotEncoder(sparse_output=False)
#     Y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
    
#     n_clases = Y_encoded.shape[1]
#     cca = CCA(n_components=n_clases)
    
#     cca.fit(X, Y_encoded)
#     X_c, Y_c = cca.transform(X, Y_encoded)
    
#     correlacion_canonica = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=n_clases)
    
#     for i, corr in enumerate(correlacion_canonica):
#         print(f"  Correlación canónica del Componente {i+1} (Características vs Clase): {corr:.4f}")

def ejecutar():
    print("\n=== INICIANDO ETAPA 3: ANÁLISIS DE CORRELACIÓN Y REDUCCIÓN ===")
    
    archivos = ['Slice_zscore.csv', 'Vh_zscore.csv']
    
    for nombre in archivos:
        archivo_entrada = os.path.join(DIR_TRANSFORMED, nombre)
        print(f"\n[Analizando {nombre}]")
        
        try:
            df = pd.read_csv(archivo_entrada, header=None)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {archivo_entrada}.")
            continue
            
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        matriz_covarianza_pearson(df, threshold=0.98)
        # reduccion_pca(X, n_componentes=10)  # Comentado a petición del usuario
        # analisis_cca(X, y)                  # Comentado a petición del usuario

if __name__ == "__main__":
    ejecutar()
