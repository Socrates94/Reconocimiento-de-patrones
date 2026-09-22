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
    
    ¿Cuándo usarlo? 
    Es ideal cuando se conocen los límites mínimos y máximos fijos (ej. píxeles de 0 a 255) 
    o cuando los algoritmos requieren que los datos estén estrictamente en un rango como [0, 1] 
    (ej. Redes Neuronales).
    
    Manejo de datos atípicos (outliers):
    Es MUY sensible a los valores atípicos. Si tienes un outlier gigante, el X_max será enorme, 
    y esto provocará que todos tus datos "normales" se compriman en un rango pequeñito cercano a 0.
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
    
    ¿Cuándo usarlo?
    Es la opción "por defecto" para muchos algoritmos como PCA, SVM o KNN. Funciona excelente 
    si los datos siguen (o se acercan) a una distribución normal (Campana de Gauss). No restringe 
    los datos a un rango exacto.
    
    Manejo de datos atípicos (outliers):
    Es afectado por los outliers, porque un valor muy extremo va a "jalar" o modificar la Media 
    y va a inflar la Desviación Estándar. Aún así, es menos catastrófico que el Min-Max.
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
    
    ¿Cuándo usarlo?
    ¡Es EL MEJOR MÉTODO cuando tu dataset está lleno de DATOS ATÍPICOS (outliers)!
    
    Manejo de datos atípicos (outliers):
    Es inmune (o robusto) a los outliers. Esto se debe a que usa la Mediana en lugar de la Media, 
    y el Rango Intercuartílico (IQR, que solo mira el 50% central de los datos) en lugar del Min y Max. 
    Un valor atípico gigante no altera ni la mediana ni el IQR, por lo que tu normalización queda limpia.
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
    
    ¿Cuándo usarlo?
    Sirve cuando simplemente queremos achicar el número moviendo el punto decimal (ej. 450 se vuelve 0.450) 
    para mantener exactamente los mismos dígitos pero a una escala menor a 1.
    
    Manejo de datos atípicos (outliers):
    También es muy afectado por los outliers. Como el valor 'j' depende del valor absoluto MÁS GRANDE (max_abs), 
    un solo valor gigante va a hacer que todos los demás valores "normales" ganen muchísimos ceros a la izquierda,
    perdiendo representatividad numérica.
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
