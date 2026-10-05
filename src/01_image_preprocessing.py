import cv2
import os
from pathlib import Path

def setup_directories():
    """Configura las rutas relativas del proyecto."""
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / 'data' / '1_raw_images'
    out_dir = base_dir / 'data' / '2_preprocessed'
    return raw_dir, out_dir

def preprocess_image(image_path, output_path):
    """Aplica el pipeline de preprocesamiento del artículo a una imagen."""
    # 0. Cargar la imagen en escala de grises (necesario para CLAHE y biometría)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Advertencia: No se pudo cargar {image_path}")
        return False
    
    # 1. Filtro de Mediana (Eliminación de ruido sal y pimienta)
    img_median = cv2.medianBlur(img, 3)
    
    # 2. Redimensionamiento a 128x128 (Interpolación bilineal)
    img_resized = cv2.resize(img_median, (128, 128), interpolation=cv2.INTER_LINEAR)
    
    # 3. Mejora de contraste (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_resized)
    
    # Guardar imagen procesada
    cv2.imwrite(str(output_path), img_clahe)
    return True

def run():
    print("======================================================")
    print("  ETAPA 1: PREPROCESAMIENTO DE IMÁGENES BIOMÉTRICAS")
    print("======================================================")
    
    raw_dir, out_dir = setup_directories()
    
    if not raw_dir.exists():
        print(f"Error: La carpeta {raw_dir} no existe.")
        return

    # Extenciones comunes en datasets biométricos
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    count = 0
    print("Iniciando escaneo y procesamiento... esto puede tomar unos minutos.\n")
    
    # rglob('*') recorre recursivamente TODAS las subcarpetas
    for img_path in raw_dir.rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in supported_extensions:
            # Replicar la estructura de subcarpetas en el directorio de salida
            relative_path = img_path.relative_to(raw_dir)
            output_path = out_dir / relative_path
            
            # Crear la subcarpeta correspondiente si no existe
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if preprocess_image(img_path, output_path):
                count += 1
                if count % 1000 == 0:
                    print(f" -> Procesadas {count} imágenes...")
                    
    print(f"\n[+] ¡Éxito! Preprocesamiento completado.")
    print(f"[+] Total de imágenes normalizadas y guardadas: {count}")
    print(f"[+] Destino: {out_dir}")

if __name__ == "__main__":
    run()
