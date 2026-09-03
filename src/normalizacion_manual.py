import pandas as pd
import numpy as np

def min_max(X):
    """
    Fórmula Min-Max:
    X_nuevo = (X - X_min) / (X_max - X_min)
    """
    # Calculamos el mínimo y máximo de cada columna
    X_min = X.min()
    X_max = X.max()
    
    # Aplicamos la fórmula matemática
    X_normalizado = (X - X_min) / (X_max - X_min)
    
    return X_normalizado

def z_score(X):
    """
    Fórmula Z-Score (StandardScaler):
    X_nuevo = (X - Media) / Desviacion_Estandar
    media = 0
    desviacion_estandar = 1
    """
    # Calculamos el promedio (media) y la desviación estándar de cada columna
    media = X.mean()
    desviacion_estandar = X.std()
    
    # Aplicamos la fórmula matemática
    X_estandarizado = (X - media) / desviacion_estandar
    
    return X_estandarizado

def robust_scaler(X):
    """
    Fórmula RobustScaler:
    X_nuevo = (X - Mediana) / Rango_Intercuartilico(Q3 - Q1)
    """
    # Calculamos la mediana (el valor justo en el medio, no se afecta por valores atípicos)
    mediana = X.median()
    
    # Calculamos el cuartil 1 (25%) y el cuartil 3 (75%)
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    
    # El rango intercuartílico (IQR) es la diferencia entre el cuartil 3 y el cuartil 1
    iqr = q3 - q1
    
    # Evitamos dividir por cero si alguna columna tiene el mismo valor en casi todas partes
    # Si IQR es 0, lo tratamos como 1 para no afectar el resultado
    iqr = iqr.replace(0, 1)
    
    # Aplicamos la fórmula matemática
    X_robusto = (X - mediana) / iqr
    
    return X_robusto

def main():
    archivos = ['Slice.txt', 'Vh.txt']
    print("--- Guardando archivos normalizados manualmente ---")
    
    for nombre_archivo in archivos:
        print(f"\nProcesando {nombre_archivo}...")
        df = pd.read_csv(nombre_archivo, header=None)
        
        # Separar características (X) y clase (y)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # 1. Min-Max
        X_minmax = min_max(X)
        df_minmax = pd.DataFrame(X_minmax)
        df_minmax[df.columns[-1]] = y.values
        out_minmax = nombre_archivo.replace('.txt', '_manual_minmax.csv')
        df_minmax.to_csv(out_minmax, index=False, header=False)
        print(f" -> Guardado: {out_minmax}")
        
        # 2. Z-Score
        X_zscore = z_score(X)
        df_zscore = pd.DataFrame(X_zscore)
        df_zscore[df.columns[-1]] = y.values
        out_zscore = nombre_archivo.replace('.txt', '_manual_zscore.csv')
        df_zscore.to_csv(out_zscore, index=False, header=False)
        print(f" -> Guardado: {out_zscore}")
        
        # 3. RobustScaler
        X_robusto = robust_scaler(X)
        df_robusto = pd.DataFrame(X_robusto)
        df_robusto[df.columns[-1]] = y.values
        out_robusto = nombre_archivo.replace('.txt', '_manual_robust.csv')
        df_robusto.to_csv(out_robusto, index=False, header=False)
        print(f" -> Guardado: {out_robusto}")

if __name__ == "__main__":
    main()
