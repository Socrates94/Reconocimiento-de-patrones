import pandas as pd
import os
from sklearn.preprocessing import KBinsDiscretizer
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_TRANSFORMED = os.path.join(BASE_DIR, 'data', '3_transformed')

def discretizar_histograma(archivo_entrada, archivo_salida, bins=5, estrategia='uniform'):
    print(f"Discretizando {os.path.basename(archivo_entrada)} (Estrategia: {estrategia}, Bins: {bins})...")
    try:
        df = pd.read_csv(archivo_entrada, header=None)
    except FileNotFoundError:
        print(f"Error: No se encontró {archivo_entrada}.")
        return
        
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    discretizador = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=estrategia)
    X_discretizado = discretizador.fit_transform(X)
    
    df_discretizado = pd.DataFrame(X_discretizado)
    df_discretizado[df.columns[-1]] = y.values
    
    df_discretizado.to_csv(archivo_salida, index=False, header=False)
    print(f" -> Guardado con éxito: {os.path.basename(archivo_salida)}")

def ejecutar():
    print("\n=== INICIANDO ETAPA 4: DISCRETIZACIÓN ===")
    archivos = ['Slice_minmax.csv', 'Vh_minmax.csv']
    
    for nombre in archivos:
        archivo_entrada = os.path.join(DIR_TRANSFORMED, nombre)
        archivo_salida = os.path.join(DIR_TRANSFORMED, nombre.replace('.csv', '_discretizado.csv'))
        
        if os.path.exists(archivo_entrada):
            discretizar_histograma(archivo_entrada, archivo_salida, bins=5, estrategia='uniform')
        else:
            print(f"Advertencia: No se encontró {archivo_entrada}. Debes ejecutar la normalización primero.")

if __name__ == "__main__":
    ejecutar()
