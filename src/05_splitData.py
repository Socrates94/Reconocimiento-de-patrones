import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_CORRELACION = os.path.join(BASE_DIR, 'data', '4_correlacion')
DIR_SPLITS = os.path.join(BASE_DIR, 'data', '5_splits')

def split_dataset(nombre_archivo, test_size=0.15, val_size=0.15):
    print(f"\n[Dividiendo {nombre_archivo}]")
    archivo_entrada = os.path.join(DIR_CORRELACION, nombre_archivo)
    
    try:
        df = pd.read_csv(archivo_entrada, header=None)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo_entrada}.")
        return

    # Extraemos X (características) y Y (etiqueta, última columna)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # 1. Primera división: Separar Train (70%) y Temp (30% para Val y Test)
    # temp_size = test_size + val_size = 0.30
    temp_size = test_size + val_size
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=42, stratify=y
    )
    
    # 2. Segunda división: Separar Temp en Validation y Test (mitad y mitad de ese 30% -> 15% y 15%)
    # val_ratio = val_size / temp_size = 0.15 / 0.30 = 0.5
    val_ratio = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    # 3. Guardar los 3 datasets reconstruidos (X + y)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    base_name = nombre_archivo.replace('.csv', '')
    
    df_train.to_csv(os.path.join(DIR_SPLITS, f"{base_name}_train.csv"), index=False, header=False)
    df_val.to_csv(os.path.join(DIR_SPLITS, f"{base_name}_val.csv"), index=False, header=False)
    df_test.to_csv(os.path.join(DIR_SPLITS, f"{base_name}_test.csv"), index=False, header=False)
    
    print(f"  -> Train (70%): {df_train.shape[0]} muestras")
    print(f"  -> Validation (15%): {df_val.shape[0]} muestras")
    print(f"  -> Test (15%): {df_test.shape[0]} muestras")

def ejecutar():
    os.makedirs(DIR_SPLITS, exist_ok=True)
    print("\n=== INICIANDO ETAPA 4: DIVISIÓN DE DATOS (TRAIN/VAL/TEST) ===")
    
    # Solo procesamos los datasets "limpios" de la etapa anterior
    if not os.path.exists(DIR_CORRELACION):
        print(f"Error: La carpeta {DIR_CORRELACION} no existe.")
        return
        
    archivos_limpios = [f for f in os.listdir(DIR_CORRELACION) if f.endswith('_limpio.csv')]
    
    if not archivos_limpios:
        print("No se encontraron datasets limpios en la carpeta 4_correlacion.")
        
    for archivo in archivos_limpios:
        split_dataset(archivo)
        
    print("\n  [+] Divisiones guardadas exitosamente en la carpeta data/5_splits/")

if __name__ == "__main__":
    ejecutar()
