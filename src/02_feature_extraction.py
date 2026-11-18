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

def process_image_gabor(img, filters, grid_size=(4, 4)):
    """
    Aplica el banco de filtros a una imagen y extrae un Vector de Características
    espacial y frecuencial.
    1. Usa cv2.CV_32F para no truncar a cero las respuestas armónicas negativas.
    2. Además de media y varianza global, divide la imagen en una rejilla espacial
       (ej. 4x4 cuadrantes) para capturar la distribución de textura en el espacio
       (preservando la topología de ojos, boca, crestas de huella y estroma de iris).
    """
    features = []
    h, w = img.shape[:2]
    gh, gw = grid_size
    step_h = h // gh
    step_w = w // gw

    for kern in filters:
        # Aplicamos la convolución con punto flotante 32-bit (conserva valores negativos)
        fimg = cv2.filter2D(img, cv2.CV_32F, kern)
        
        # 1. Estadísticas globales
        features.extend([float(fimg.mean()), float(fimg.var())])
        
        # 2. Estadísticas locales espaciales por bloques
        for i in range(gh):
            for j in range(gw):
                block = fimg[i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w]
                features.extend([float(block.mean()), float(block.var())])
        
    # Devolvemos un arreglo unidimensional de NumPy (El Embedding espacial-frecuencial)
    return np.array(features, dtype=np.float32)

from concurrent.futures import ProcessPoolExecutor, as_completed

def _worker_extract(task):
    img_path, output_path, filters = task
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        feature_vector = process_image_gabor(img, filters)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), feature_vector)
        return True
    except Exception as e:
        return False

def extract_features(image_path, output_path, filters):
    """Carga la imagen, extrae características y guarda el vector."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Advertencia: No se pudo cargar {image_path}")
        return False
    
    # 1. Pasar por Gabor con cuadrícula espacial
    feature_vector = process_image_gabor(img, filters)
    
    # 2. Guardar como binario de NumPy (.npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), feature_vector)
    return True

def run(max_workers=None):
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
    print(f"[+] Rejilla espacial activa: 4x4 cuadrantes (408 dimensiones por vector).")
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    tasks = []
    
    print("\nRecolectando lista de imágenes para procesamiento en paralelo...")
    for img_path in proc_dir.rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in supported_extensions:
            relative_path = img_path.relative_to(proc_dir)
            output_path = feat_dir / relative_path.with_suffix('.npy')
            tasks.append((img_path, output_path, filters))
            
    total_tasks = len(tasks)
    print(f"[+] Total de imágenes encontradas: {total_tasks}")
    print(f"[+] Ejecutando extracción matemática con ProcessPoolExecutor (todos los núcleos)...")
    
    count = 0
    # Procesamiento paralelo eficiente
    workers = max_workers or max(1, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for ok in executor.map(_worker_extract, tasks, chunksize=100):
            if ok:
                count += 1
                if count % 1000 == 0 or count == total_tasks:
                    print(f" -> Características extraídas de {count}/{total_tasks} imágenes...")
                    
    print(f"\n[+] ¡Éxito! Extracción Gabor completada.")
    print(f"[+] Total de embeddings (vectores) creados: {count}")
    print(f"[+] Destino: {feat_dir}")

if __name__ == "__main__":
    run()
