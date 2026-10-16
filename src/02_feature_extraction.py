import cv2
import numpy as np
import os
from pathlib import Path

def setup_directories():
    """Configura las rutas relativas del proyecto."""
    base_dir = Path(__file__).resolve().parent.parent
    proc_dir = base_dir / 'data' / '2_preprocessed'
    feat_dir = base_dir / 'data' / '3_features'
    return proc_dir, feat_dir

def build_filters():
    """
    Construye un banco de filtros Gabor matemáticos.
    Variamos la orientación (theta) y la frecuencia (lambda) para capturar 
    arrugas finas, gruesas y en distintos ángulos (horizontal, vertical, diagonal).
    """
    filters = []
    ksize = 31 # Tamaño del kernel del filtro
    
    # 4 orientaciones: 0, 45, 90, 135 grados
    for theta in np.arange(0, np.pi, np.pi / 4): 
        # 3 frecuencias distintas
        for lamda in np.arange(np.pi / 4, np.pi, np.pi / 4): 
            # Fórmula de Gabor de OpenCV
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            # Normalización del kernel
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters

def process_image_gabor(img, filters):
    """
    Aplica el banco de filtros a una imagen y extrae un Vector de Características.
    Por cada filtro, sacamos la media y la varianza de la textura detectada.
    """
    features = []
    for kern in filters:
        # Aplicamos la convolución del filtro sobre la imagen
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        
        # Extraemos la estadística matemática de esa textura
        mean = fimg.mean()
        var = fimg.var()
        features.extend([mean, var])
        
    # Devolvemos un arreglo unidimensional de NumPy (El Embedding)
    return np.array(features, dtype=np.float32)

def extract_features(image_path, output_path, filters):
    """Carga la imagen, extrae características y guarda el vector."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Advertencia: No se pudo cargar {image_path}")
        return False
    
    # 1. Pasar por Gabor
    feature_vector = process_image_gabor(img, filters)
    
    # 2. Guardar como binario de NumPy (.npy)
    # Ya no guardamos una foto, guardamos la matemática pura.
    np.save(str(output_path), feature_vector)
    return True

def run():
    print("======================================================")
    print("  ETAPA 2: EXTRACCIÓN DE CARACTERÍSTICAS (GABOR)")
    print("======================================================")
    
    proc_dir, feat_dir = setup_directories()
    
    if not proc_dir.exists():
        print(f"Error: La carpeta {proc_dir} no existe. Ejecuta la Etapa 1 primero.")
        return

    print("Construyendo el banco de filtros Gabor...")
    filters = build_filters()
    print(f"[+] Banco creado con {len(filters)} filtros espaciales distintos.")
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    print("\nIniciando extracción matemática... (Esto usará bastante CPU).")
    
    for img_path in proc_dir.rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in supported_extensions:
            # Replicar la estructura de carpetas
            relative_path = img_path.relative_to(proc_dir)
            
            # ATENCIÓN: Ya no usamos extensiones de imagen (.jpg). Ahora son vectores (.npy)
            output_path = feat_dir / relative_path.with_suffix('.npy')
            
            # Crear subcarpeta si no existe
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if extract_features(img_path, output_path, filters):
                count += 1
                if count % 1000 == 0:
                    print(f" -> Características extraídas de {count} imágenes...")
                    
    print(f"\n[+] ¡Éxito! Extracción Gabor completada.")
    print(f"[+] Total de embeddings (vectores) creados: {count}")
    print(f"[+] Destino: {feat_dir}")

if __name__ == "__main__":
    run()
