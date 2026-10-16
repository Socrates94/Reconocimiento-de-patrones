import os
import numpy as np
from pathlib import Path
import pygad
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Ignoramos warnings de sklearn para no ensuciar la consola durante la evolución
warnings.filterwarnings('ignore')

# Variables globales para que PyGAD pueda verlas en la función de fitness
X_train_global, X_test_global, y_train_global, y_test_global = None, None, None, None

def setup_directories():
    base_dir = Path(__file__).resolve().parent.parent
    feat_dir = base_dir / 'data' / '3_features'
    opt_dir = base_dir / 'data' / '4_optimized'
    opt_dir.mkdir(parents=True, exist_ok=True)
    return feat_dir, opt_dir

def fitness_function(ga_instance, solution, solution_idx):
    """ 
    Función objetivo (Fitness) que el Algoritmo Genético intentará maximizar. 
    Representa el 'ambiente' al que deben sobrevivir las características.
    """
    global X_train_global, X_test_global, y_train_global, y_test_global
    
    # Extraer los índices donde el gen es 1 (características seleccionadas)
    selected_features = np.where(solution == 1)[0]
    
    # Si la mutación apagó todos los genes, el cromosoma muere (fitness = 0)
    if len(selected_features) == 0:
        return 0.0
    
    # Reducimos el vector Gabor masivo usando solo los índices seleccionados
    X_train_reduced = X_train_global[:, selected_features]
    X_test_reduced = X_test_global[:, selected_features]
    
    # Entrenamos un SVM rápido en memoria para probar estas características
    # max_iter limita las iteraciones matemáticas para no congelar el procesador
    clf = SVC(kernel='linear', max_iter=200) 
    clf.fit(X_train_reduced, y_train_global)
    preds = clf.predict(X_test_reduced)
    
    accuracy = accuracy_score(y_test_global, preds)
    
    # RECOMPENSA: Alta exactitud. PENALIZACIÓN: Usar muchas características.
    # Queremos que la IA busque el porcentaje más alto con la menor cantidad de números.
    fitness = accuracy - (0.0001 * len(selected_features))
    return fitness

def load_modality_data(modality_path):
    """ Recorre la modalidad (cara, huella o iris) y junta los .npy en una gran matriz. """
    X = []
    y = []
    
    classes = [d for d in modality_path.iterdir() if d.is_dir()]
    for class_idx, class_dir in enumerate(classes):
        for npy_file in class_dir.rglob('*.npy'):
            features = np.load(str(npy_file))
            X.append(features)
            y.append(class_idx) # La etiqueta es el índice de la clase
            
    return np.array(X), np.array(y)

def optimize_features(feat_dir, opt_dir):
    global X_train_global, X_test_global, y_train_global, y_test_global
    
    modalities = ['face', 'fingerprint', 'iris']
    
    for mod in modalities:
        mod_path = feat_dir / mod
        if not mod_path.exists():
            continue
            
        print(f"\n--- Optimizando características para: {mod.upper()} ---")
        X, y = load_modality_data(mod_path)
        
        if len(X) == 0:
            print(f"[-] No se encontraron datos de {mod}.")
            continue
            
        num_features = X.shape[1]
        print(f"[+] Cargado: {X.shape[0]} muestras con {num_features} dimensiones Gabor.")
        
        # Subdividimos los datos temporalmente para que el GA tenga un campo de pruebas
        X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Configuración del Algoritmo Genético (PyGAD)
        # Ajustado a números pequeños para demostración (aumentar generaciones en el entrenamiento final)
        ga_instance = pygad.GA(
            num_generations=5,             # Ciclos de evolución (5 es rápido)
            num_parents_mating=4,          # Los mejores 4 se cruzan
            fitness_func=fitness_function,
            sol_per_pop=10,                # Población de 10 "Bacterias/Individuos"
            num_genes=num_features,        # El cromosoma mide lo mismo que el vector Gabor
            gene_type=int,                 # Sistema Binario
            gene_space=[0, 1],             # 0 = eliminar característica, 1 = conservarla
            mutation_percent_genes=5       # Tasa de mutación del 5%
        )
        
        print("Iniciando la evolución darwiniana de las características...")
        ga_instance.run()
        
        # Extraemos al individuo alfa (la mejor combinación)
        solution, solution_fitness, _ = ga_instance.best_solution()
        selected_features = np.where(solution == 1)[0]
        
        print(f"[!] Evolución Finalizada. Fitness ganador: {solution_fitness:.4f}")
        print(f"[!] Dimensionalidad destruida: de {num_features} a solo {len(selected_features)} características.")
        
        # 1. Guardamos los índices que ganaron por si queremos verlos después
        np.save(str(opt_dir / f'mask_{mod}.npy'), selected_features)
        
        # 2. Re-generamos la matriz X pero aplastando todo lo inútil
        X_optimized = X[:, selected_features]
        np.save(str(opt_dir / f'X_{mod}_opt.npy'), X_optimized)
        np.save(str(opt_dir / f'y_{mod}.npy'), y)
        print(f"[+] Matrices depuradas guardadas en {opt_dir}")

def run():
    print("======================================================")
    print("  ETAPA 3: SELECCIÓN Y OPTIMIZACIÓN (ALGORITMOS GENÉTICOS)")
    print("======================================================")
    feat_dir, opt_dir = setup_directories()
    optimize_features(feat_dir, opt_dir)

if __name__ == "__main__":
    run()
