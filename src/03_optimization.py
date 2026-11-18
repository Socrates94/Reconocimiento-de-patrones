import os
import numpy as np
from pathlib import Path
import pygad
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
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

def evaluate_chromosome(solution, X_tr, y_tr, X_te, y_te):
    """Evalúa la aptitud de un vector binario de características."""
    selected_features = np.where(solution == 1)[0]
    if len(selected_features) == 0:
        return 0.0
    
    X_tr_sub = X_tr[:, selected_features]
    X_te_sub = X_te[:, selected_features]
    
    # Estandarización imprescindible para el cálculo del margen euclidiano del SVM
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_sub)
    X_te_scaled = scaler.transform(X_te_sub)
    
    # LinearSVC optimizado para clasificación rápida
    clf = LinearSVC(dual='auto', max_iter=600, random_state=42)
    clf.fit(X_tr_scaled, y_tr)
    preds = clf.predict(X_te_scaled)
    acc = accuracy_score(y_te, preds)
    
    # Función de aptitud multiobjetivo del artículo:
    # Fitness = w1 * Accuracy + w2 * (1 - k/D)
    w1, w2 = 0.90, 0.10
    compaction = 1.0 - (len(selected_features) / len(solution))
    return float(w1 * acc + w2 * compaction)

def fitness_function(ga_instance, solution, solution_idx):
    """Función objetivo que el Algoritmo Genético maximiza."""
    global X_train_global, X_test_global, y_train_global, y_test_global
    return evaluate_chromosome(solution, X_train_global, y_train_global, X_test_global, y_test_global)

def bfo_local_refinement(best_solution, X_tr, y_tr, X_te, y_te, max_chemotaxis_steps=25):
    """
    Bacterial Foraging Optimization (BFO) - Quimiotaxis local.
    Refina la mejor solución de GA ejecutando pequeñas perturbaciones (voltereta y nado)
    para pulir la selección en el espacio combinatorio de características.
    """
    current_sol = np.array(best_solution, dtype=int).copy()
    current_fit = evaluate_chromosome(current_sol, X_tr, y_tr, X_te, y_te)
    
    rng = np.random.RandomState(42)
    n_genes = len(current_sol)
    
    for _ in range(max_chemotaxis_steps):
        # Voltereta (Tumble): altera 1 a 2 genes al azar
        cand = current_sol.copy()
        k_flips = rng.randint(1, 3)
        flip_indices = rng.choice(n_genes, size=k_flips, replace=False)
        cand[flip_indices] = 1 - cand[flip_indices]
        
        if cand.sum() == 0:
            continue
            
        cand_fit = evaluate_chromosome(cand, X_tr, y_tr, X_te, y_te)
        # Nado (Swim): avanza si la concentración de nutrientes (fitness) aumenta
        if cand_fit > current_fit:
            current_sol = cand
            current_fit = cand_fit
            
    return current_sol, current_fit

def load_face_data(modality_path):
    """
    FACE: clasificación binaria real (1) vs falsa (0).
    Usa 'real_and_fake_face' para evitar duplicados.
    """
    X, y = [], []
    base = modality_path / 'real_and_fake_face'
    if not base.exists():
        candidates = [d for d in modality_path.iterdir() if d.is_dir()]
        if not candidates:
            return np.array(X), np.array(y)
        base = sorted(candidates)[0]

    label_map = {'training_real': 1, 'training_fake': 0,
                 'real': 1, 'fake': 0}
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        label = None
        for key, val in label_map.items():
            if key in subdir.name.lower():
                label = val
                break
        if label is None:
            continue
        for npy_file in sorted(subdir.rglob('*.npy')):
            X.append(np.load(str(npy_file)))
            y.append(label)
    return np.array(X), np.array(y)

def load_fingerprint_data(modality_path, max_per_class=6000):
    """
    FINGERPRINT: clasificación binaria Real/Genuino (1) vs Alterada/Ataque (0).
    Equilibra el dataset tomando un subconjunto balanceado para erradicar el
    colapso del clasificador por desbalance extremo (89% alteradas).
    """
    X_real, y_real = [], []
    X_alt, y_alt = [], []
    
    base = modality_path / 'SOCOFing'
    if not base.exists():
        candidates = [d for d in modality_path.iterdir() if d.is_dir()]
        if not candidates:
            return np.array([]), np.array([])
        base = sorted(candidates)[0]

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        name_lower = subdir.name.lower()
        if name_lower == 'real':
            for npy_file in sorted(subdir.rglob('*.npy')):
                X_real.append(np.load(str(npy_file)))
                y_real.append(1)  # 1 = Genuino
        elif name_lower == 'altered':
            for npy_file in sorted(subdir.rglob('*.npy')):
                X_alt.append(np.load(str(npy_file)))
                y_alt.append(0)  # 0 = Alterada

    X_real = np.array(X_real)
    y_real = np.array(y_real)
    X_alt = np.array(X_alt)
    y_alt = np.array(y_alt)

    # Balanceo determinista 50/50
    target_count = min(len(X_real), len(X_alt), max_per_class)
    if target_count > 0:
        rng = np.random.RandomState(42)
        if len(X_alt) > target_count:
            idx_alt = rng.choice(len(X_alt), size=target_count, replace=False)
            X_alt = X_alt[idx_alt]
            y_alt = y_alt[idx_alt]
            
        if len(X_real) > target_count:
            idx_real = rng.choice(len(X_real), size=target_count, replace=False)
            X_real = X_real[idx_real]
            y_real = y_real[idx_real]

    if len(X_real) == 0 or len(X_alt) == 0:
        return np.array([]), np.array([])

    X = np.vstack([X_real, X_alt])
    y = np.concatenate([y_real, y_alt])
    
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]

def load_iris_data(modality_path):
    """
    IRIS: identificación multiclase por persona (sujetos de MMU Iris Database).
    """
    X, y = [], []
    base = modality_path / 'MMU-Iris-Database'
    if not base.exists():
        candidates = [d for d in modality_path.iterdir() if d.is_dir()]
        if not candidates:
            return np.array(X), np.array(y)
        base = sorted(candidates)[0]

    person_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        key=lambda d: int(d.name) if d.name.isdigit() else 0
    )
    for class_idx, person_dir in enumerate(person_dirs):
        for npy_file in sorted(person_dir.rglob('*.npy')):
            X.append(np.load(str(npy_file)))
            y.append(class_idx)
    return np.array(X), np.array(y)

def load_modality_data(modality_path, modality_name):
    loaders = {
        'face':        load_face_data,
        'fingerprint': load_fingerprint_data,
        'iris':        load_iris_data,
    }
    loader = loaders.get(modality_name)
    if loader is None:
        return np.array([]), np.array([])
    return loader(modality_path)

def build_multimodal_dataset(opt_dir, total_samples=2330):
    """
    Construye el dataset multi-biométrico quimérico (Face + Fingerprint + Iris)
    fusionando las representaciones a nivel de características (feature-level fusion).
    Garantiza que el 30% de test blind sea exactamente N = 699 muestras, replicando
    el tamaño exacto del artículo.
    """
    print("\n--- Generando Dataset Multi-biométrico Combinado (Fusión) ---")
    face_x_p = opt_dir / 'X_face_opt.npy'
    face_y_p = opt_dir / 'y_face.npy'
    fp_x_p   = opt_dir / 'X_fingerprint_opt.npy'
    fp_y_p   = opt_dir / 'y_fingerprint.npy'
    iris_x_p = opt_dir / 'X_iris_opt.npy'
    iris_y_p = opt_dir / 'y_iris.npy'
    
    if not (face_x_p.exists() and fp_x_p.exists() and iris_x_p.exists()):
        print("[-] Faltan matrices optimizadas individuales para construir la fusión multimodal.")
        return
        
    X_face, y_face = np.load(str(face_x_p)), np.load(str(face_y_p))
    X_fp, y_fp     = np.load(str(fp_x_p)), np.load(str(fp_y_p))
    X_iris, y_iris = np.load(str(iris_x_p)), np.load(str(iris_y_p))
    
    # Índices genuinos (1) y no genuinos (0)
    idx_face_gen = np.where(y_face == 1)[0]
    idx_face_imp = np.where(y_face == 0)[0]
    idx_fp_gen   = np.where(y_fp == 1)[0]
    idx_fp_imp   = np.where(y_fp == 0)[0]
    
    # En el artículo: Test set N=699 (458 genuinos / 241 no-genuinos).
    # Dataset total (N=2330): ~1527 genuinos y ~803 impostores (proporción 65.5% / 34.5%).
    n_gen = int(round(total_samples * (458 / 699)))
    n_imp = total_samples - n_gen
    
    rng = np.random.RandomState(42)
    
    X_multi, y_multi = [], []
    
    # 1. Crear perfiles Genuinos (Rostro real + Huella real + Iris del sujeto)
    for i in range(n_gen):
        f_idx = idx_face_gen[i % len(idx_face_gen)]
        fp_idx = idx_fp_gen[i % len(idx_fp_gen)]
        iris_idx = i % len(X_iris)
        
        vec = np.concatenate([X_face[f_idx], X_fp[fp_idx], X_iris[iris_idx]])
        X_multi.append(vec)
        y_multi.append(1)
        
    # 2. Crear perfiles Impostores/Ataques (Rostro falso o Huella alterada o combinación de ataque)
    for j in range(n_imp):
        f_idx = idx_face_imp[j % len(idx_face_imp)]
        fp_idx = idx_fp_imp[j % len(idx_fp_imp)]
        # Iris cruzado (sujeto no coincidente)
        iris_idx = rng.randint(0, len(X_iris))
        
        vec = np.concatenate([X_face[f_idx], X_fp[fp_idx], X_iris[iris_idx]])
        X_multi.append(vec)
        y_multi.append(0)
        
    X_multi = np.array(X_multi, dtype=np.float32)
    y_multi = np.array(y_multi, dtype=int)
    
    perm = rng.permutation(len(X_multi))
    X_multi, y_multi = X_multi[perm], y_multi[perm]
    
    np.save(str(opt_dir / 'X_multimodal_opt.npy'), X_multi)
    np.save(str(opt_dir / 'y_multimodal.npy'), y_multi)
    print(f"[+] Dataset Multi-biométrico generado: {X_multi.shape[0]} muestras con {X_multi.shape[1]} dimensiones fusionadas.")
    print(f"[+] Proporción: {n_gen} Genuinos (1) y {n_imp} Impostores/Ataques (0).")

def optimize_features(feat_dir, opt_dir):
    global X_train_global, X_test_global, y_train_global, y_test_global
    
    modalities = ['face', 'fingerprint', 'iris']
    
    for mod in modalities:
        mod_path = feat_dir / mod
        if not mod_path.exists():
            continue
            
        print(f"\n--- Optimizando características para: {mod.upper()} ---")
        X, y = load_modality_data(mod_path, mod)
        
        if len(X) == 0:
            print(f"[-] No se encontraron datos de {mod}.")
            continue
            
        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        print(f"[+] Cargado: {X.shape[0]} muestras con {num_features} dimensiones Gabor ({num_classes} clases).")

        if num_classes < 2:
            print(f"[-] Solo se detectó {num_classes} clase en {mod}. Saltando...")
            continue

        # Submuestreo para optimización rápida si el dataset es muy grande
        max_opt_samples = 4000
        if len(X) > max_opt_samples:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(len(X), size=max_opt_samples, replace=False)
            X_sub, y_sub = X[sample_idx], y[sample_idx]
        else:
            X_sub, y_sub = X, y

        # Split estratificado para evaluación de fitness durante la optimización
        min_class_count = min(np.bincount(y_sub))
        use_stratify = y_sub if min_class_count >= 2 else None

        X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
            X_sub, y_sub, test_size=0.25, stratify=use_stratify, random_state=42
        )
        
        # Configuración del Algoritmo Genético (PyGAD) con parámetros robustos
        ga_instance = pygad.GA(
            num_generations=12,
            num_parents_mating=6,
            fitness_func=fitness_function,
            sol_per_pop=15,
            num_genes=num_features,
            gene_type=int,
            gene_space=[0, 1],
            mutation_percent_genes=8,
            random_seed=42
        )
        
        print(f"[+] Iniciando evolución genética (GA) sobre {num_features} dimensiones...")
        ga_instance.run()
        
        best_sol, best_fitness, _ = ga_instance.best_solution()
        print(f"[!] GA finalizado. Fitness alcanzado: {best_fitness:.4f}")
        
        # Refinamiento Bioinspirado Local: Bacterial Foraging Optimization (BFO)
        print("[+] Aplicando refinamiento quimiotáctico BFO (Bacterial Foraging Optimization)...")
        bfo_sol, bfo_fitness = bfo_local_refinement(
            best_sol, X_train_global, y_train_global, X_test_global, y_test_global, max_chemotaxis_steps=20
        )
        print(f"[!] BFO completado. Fitness optimizado: {bfo_fitness:.4f}")
        
        selected_features = np.where(bfo_sol == 1)[0]
        # Asegurar al menos una característica seleccionada
        if len(selected_features) == 0:
            selected_features = np.arange(min(num_features, 10))
            
        print(f"[!] Selección final (GA+BFO): de {num_features} a {len(selected_features)} características óptimas.")
        
        np.save(str(opt_dir / f'mask_{mod}.npy'), selected_features)
        X_optimized = X[:, selected_features]
        np.save(str(opt_dir / f'X_{mod}_opt.npy'), X_optimized)
        np.save(str(opt_dir / f'y_{mod}.npy'), y)
        print(f"[+] Matrices depuradas guardadas en {opt_dir}")
        
    # Paso de integración: Construir el dataset multimodal combinado
    build_multimodal_dataset(opt_dir)

def run():
    print("=================================================================")
    print("  ETAPA 3: SELECCIÓN Y OPTIMIZACIÓN HÍBRIDA (GA + BFO)")
    print("=================================================================")
    feat_dir, opt_dir = setup_directories()
    optimize_features(feat_dir, opt_dir)

if __name__ == "__main__":
    run()
