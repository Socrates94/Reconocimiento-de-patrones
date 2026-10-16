import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')

def setup_directories():
    base_dir = Path(__file__).resolve().parent.parent
    opt_dir = base_dir / 'data' / '4_optimized'
    split_dir = base_dir / 'data' / '5_splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    return opt_dir, split_dir

def train_hybrid_model(X_train, y_train):
    """
    Construye y entrena el ensamble matemático combinando la frontera rígida
    del SVM con la resistencia al ruido del Bosque Aleatorio.
    """
    # 1. Máquina de Vectores de Soporte (SVM)
    # Se activa probability=True para que el ensamble pueda promediar las certezas matemáticas
    svm_clf = SVC(kernel='linear', probability=True, random_state=42)
    
    # 2. Bosque Aleatorio (RF)
    # 🚀 ACTIVACIÓN DEL HARDWARE EXTREMO: n_jobs=-1 pone a trabajar a los 24 núcleos
    rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    
    # 3. El Sistema Híbrido (Voting Classifier)
    # Usa "soft voting": Si SVM dice 90% seguro que es real y RF dice 80%, promedia a 85%.
    hybrid_clf = VotingClassifier(
        estimators=[('svm', svm_clf), ('rf', rf_clf)],
        voting='soft',
        n_jobs=-1 # También permitimos que ambos modelos se entrenen en paralelo simultáneamente
    )
    
    hybrid_clf.fit(X_train, y_train)
    return hybrid_clf

def run():
    print("======================================================")
    print("  ETAPA 4: CLASIFICADOR HÍBRIDO (SVM + RF)")
    print("======================================================")
    
    opt_dir, split_dir = setup_directories()
    modalities = ['face', 'fingerprint', 'iris']
    count_models = 0
    
    for mod in modalities:
        x_path = opt_dir / f'X_{mod}_opt.npy'
        y_path = opt_dir / f'y_{mod}.npy'
        
        if not x_path.exists() or not y_path.exists():
            continue
            
        print(f"\n--- Entrenando modelo para: {mod.upper()} ---")
        X = np.load(str(x_path))
        y = np.load(str(y_path))
        
        # 1. División de datos según el artículo: 70% Train, 30% Test
        print("[+] Dividiendo dataset (70% Train / 30% Test) de forma estratificada...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        
        # Guardamos las porciones de Test intactas para que la Etapa 5 pueda usarlas como examen final
        np.save(str(split_dir / f'X_test_{mod}.npy'), X_test)
        np.save(str(split_dir / f'y_test_{mod}.npy'), y_test)
        
        # 2. Entrenamiento Masivo Paralelo
        print(f"[+] Entrenando Híbrido SVM-RF con {X_train.shape[0]} muestras...")
        print("[!] Forzando multi-procesamiento en CPU (n_jobs=-1)...")
        hybrid_model = train_hybrid_model(X_train, y_train)
        
        # 3. Serialización y Guardado del modelo entrenado
        model_path = split_dir / f'hybrid_model_{mod}.joblib'
        joblib.dump(hybrid_model, str(model_path))
        print(f"[+] Modelo físico compilado y guardado exitosamente: {model_path.name}")
        count_models += 1
        
    if count_models > 0:
        print("\n[+] ¡Éxito! Todos los modelos han sido entrenados y empaquetados.")
    else:
        print("\n[-] No se encontraron datos optimizados. Por favor corre la Etapa 3 primero.")

if __name__ == "__main__":
    run()
