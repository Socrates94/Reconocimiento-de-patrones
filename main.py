import sys
import os
import importlib

# Aseguramos que la carpeta src sea reconocida como módulo
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# En Python, los módulos no pueden empezar con números al importarse directamente.
# Es por eso que usamos importlib para cargarlos dinámicamente.
proc = importlib.import_module("01_procesamiento")
norm = importlib.import_module("02_normalizacion")
ana = importlib.import_module("03_analisis")
disc = importlib.import_module("04_discretizacion")

def main():
    print("======================================================")
    print("  PIPELINE DE RECONOCIMIENTO DE PATRONES - TAREA 01")
    print("======================================================\n")
    
    # 1. Convertir de TXT a CSV
    proc.ejecutar()
    
    # 2. Normalizar (Z-Score, Min-Max, Robust)
    norm.ejecutar()
    
    # 3. Discretizar los datos Min-Max
    disc.ejecutar()
    
    # 4. Análisis Exploratorio Matemático (PCA, CCA, Pearson)
    ana.ejecutar()
    
    print("\n======================================================")
    print("  PIPELINE COMPLETADO EXITOSAMENTE")
    print("  Los datos están listos en la carpeta data/3_transformed/")
    print("======================================================")

if __name__ == "__main__":
    main()
