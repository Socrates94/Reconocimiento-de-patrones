import csv
import os

# Rutas dinámicas basadas en la carpeta donde está este script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RAW = os.path.join(BASE_DIR, 'data', '1_raw')
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', '2_processed')

def process_file(input_file, output_csv):
    print(f"Convirtiendo {os.path.basename(input_file)} a formato CSV para poderlo leer mas claro...")
    
    # Simplemente leemos el archivo original y lo escribimos de nuevo sin alterar el orden
    with open(input_file, 'r') as f_in, open(output_csv, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # Leemos cada fila y la escribimos tal cual está
        for row in reader:
            writer.writerow(row)
            
    print(f" -> Guardado como {os.path.basename(output_csv)}\n")

def ejecutar():
    print("=== INICIANDO ETAPA 1: PREPROCESAMIENTO ===")
    for name in ['Slice', 'Vh']:
        archivo_entrada = os.path.join(DIR_RAW, f'{name}.txt')
        archivo_salida = os.path.join(DIR_PROCESSED, f'{name}.csv')
        
        if os.path.exists(archivo_entrada):
            process_file(archivo_entrada, archivo_salida)
        else:
            print(f"Advertencia: No se encontró el archivo original {archivo_entrada}")

if __name__ == "__main__":
    ejecutar()
