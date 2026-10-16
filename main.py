# Orquestador del Proyecto Final Biomédico
# Este archivo será el encargado de ejecutar el pipeline paso a paso

import importlib

# Importamos los módulos usando importlib
etapa1 = importlib.import_module("src.01_image_preprocessing")
etapa2 = importlib.import_module("src.02_feature_extraction")
etapa3 = importlib.import_module("src.03_optimization")
etapa4 = importlib.import_module("src.04_model_training")
etapa5 = importlib.import_module("src.05_evaluation")

def main():
    # Descomenta SOLO la etapa que necesites correr. 
    # El flujo natural es ejecutar desde la 1 hasta la 5 en orden.
    
    # etapa1.run() # Lee imágenes crudas, aplica filtro mediana, resize y CLAHE
    # etapa2.run() # Lee imágenes procesadas y extrae vectores Gabor
    # etapa3.run() # Toma vectores masivos, aplica GA (Alg. Genético) y los reduce
    # etapa4.run() # Parte datos (70/30), entrena SVM-RF híbrido y guarda en disco
    etapa5.run()   # Lee modelo e imprime métricas, curvas ROC y matriz de confusión

if __name__ == "__main__":
    main()
