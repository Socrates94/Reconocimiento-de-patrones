import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RAW = os.path.join(BASE_DIR, 'data', '1_raw')
DIR_TRANSFORMED = os.path.join(BASE_DIR, 'data', '3_transformed')

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

def decimal_scaling(X):
    """
    Fórmula Decimal Scaling:
    X_nuevo = X / (10^j)
    donde j es la cantidad de veces que se recorre el punto para que el máximo absoluto sea < 1
    """
    # 1. Encontrar el valor absoluto máximo por cada columna
    max_abs = np.abs(X).max()
    
    # Prevenimos el error matemático de log10(0) si alguna columna tiene puros ceros
    max_abs = max_abs.replace(0, 1)
    
    # 2. Calcular 'j' (el número de dígitos enteros)
    # Ej: max=250 -> log10(250)=2.39 -> ceil(2.39)=3 -> 10^3 = 1000.
    j = np.ceil(np.log10(max_abs))
    
    # 3. Aplicar la división recorriendo el punto decimal
    X_escalado = X / (10 ** j)
    
    return X_escalado

def main():
    archivos = ['Slice', 'Vh']
    print("--- Guardando archivos normalizados manualmente ---")
    
    for nombre in archivos:
        archivo_entrada = os.path.join(DIR_RAW, f'{nombre}.txt')
        if not os.path.exists(archivo_entrada):
            print(f"No se encontró {archivo_entrada}")
            continue
            
        print(f"\nProcesando {nombre}.txt...")
        df = pd.read_csv(archivo_entrada, header=None)
        
        # Separar características (X) y clase (y)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # 1. Min-Max
        X_minmax = min_max(X)
        df_minmax = pd.DataFrame(X_minmax)
        df_minmax[df.columns[-1]] = y.values
        out_minmax = os.path.join(DIR_TRANSFORMED, f'{nombre}_manual_minmax.csv')
        df_minmax.to_csv(out_minmax, index=False, header=False)
        print(f" -> Guardado: {os.path.basename(out_minmax)}")
        
        # 2. Z-Score
        X_zscore = z_score(X)
        df_zscore = pd.DataFrame(X_zscore)
        df_zscore[df.columns[-1]] = y.values
        out_zscore = os.path.join(DIR_TRANSFORMED, f'{nombre}_manual_zscore.csv')
        df_zscore.to_csv(out_zscore, index=False, header=False)
        print(f" -> Guardado: {os.path.basename(out_zscore)}")
        
        # 3. RobustScaler
        X_robusto = robust_scaler(X)
        df_robusto = pd.DataFrame(X_robusto)
        df_robusto[df.columns[-1]] = y.values
        out_robusto = os.path.join(DIR_TRANSFORMED, f'{nombre}_manual_robust.csv')
        df_robusto.to_csv(out_robusto, index=False, header=False)
        print(f" -> Guardado: {os.path.basename(out_robusto)}")
        
        # 4. Decimal Scaling (NUEVO)
        X_decimal = decimal_scaling(X)
        df_decimal = pd.DataFrame(X_decimal)
        df_decimal[df.columns[-1]] = y.values
        out_decimal = os.path.join(DIR_TRANSFORMED, f'{nombre}_manual_decimal.csv')
        df_decimal.to_csv(out_decimal, index=False, header=False)
        print(f" -> Guardado: {os.path.basename(out_decimal)}")

if __name__ == "__main__":
    main()
