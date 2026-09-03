import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RAW = os.path.join(BASE_DIR, 'data', '1_raw')
DIR_TRANSFORMED = os.path.join(BASE_DIR, 'data', '3_transformed')

def normalizar(archivo_entrada, archivo_salida, metodo='minmax'):
    
    print(f"Procesando {os.path.basename(archivo_entrada)} con el método {metodo}...")
    
    # 1. Leer el archivo txt (pandas asume que está separado por comas)
    df = pd.read_csv(archivo_entrada, header=None)
    
    # 2. Separar características (X) y la clase (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # 3. Inicializar el normalizador
    if metodo == 'minmax':
        scaler = MinMaxScaler()
    elif metodo == 'zscore':
        scaler = StandardScaler()
    elif metodo == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Método no reconocido. Usa 'minmax', 'zscore' o 'robust'.")
    
    # 4. Normalizar solo las características (X)
    X_normalizado = scaler.fit_transform(X)
    df_normalizado = pd.DataFrame(X_normalizado)
    
    # 5. Volver a pegarle la columna de la clase al final
    df_normalizado[df.columns[-1]] = y.values
    
    # 6. Guardar el resultado en la carpeta transformed
    df_normalizado.to_csv(archivo_salida, index=False, header=False)
    
    print(f" -> ¡Éxito! Guardado como {os.path.basename(archivo_salida)}\n")

def ejecutar():
    
    print("=== INICIANDO ETAPA 2: NORMALIZACION ===")
    
    metodos = ['minmax', 'zscore', 'robust']
    archivos = ['Slice', 'Vh']
    
    for name in archivos:
        
        archivo_entrada = os.path.join(DIR_RAW, f'{name}.txt')
        if not os.path.exists(archivo_entrada):
            print(f"Advertencia: No se encontró el archivo original {archivo_entrada}")
            continue
            
        for m in metodos:
            archivo_salida = os.path.join(DIR_TRANSFORMED, f'{name}_{m}.csv')
            normalizar(archivo_entrada, archivo_salida, metodo=m)

if __name__ == "__main__":
    ejecutar()
