# Orquestador del Proyecto Final Biomédico
# Este archivo será el encargado de ejecutar el pipeline paso a paso

import importlib

# Importamos el módulo que empieza con número usando importlib
etapa1 = importlib.import_module("src.01_image_preprocessing")

def main():
    etapa1.run()

if __name__ == "__main__":
    main()
