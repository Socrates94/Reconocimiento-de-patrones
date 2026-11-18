import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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

def build_models():
    """
    Construye las arquitecturas de clasificación con estandarización (StandardScaler):
    1. SVM con kernel lineal y calibración sigmoidal de probabilidades.
    2. Random Forest (100 estimadores) para mitigación de ruido y no linealidad.
    3. Híbrido SVM-RF con votación suave (soft voting).
    """
    # 1. Pipeline SVM Lineal con StandardScaler y Calibración
    linear_svc = LinearSVC(dual='auto', max_iter=4000, random_state=42, class_weight='balanced')
    calibrated_svm = CalibratedClassifierCV(linear_svc, cv=3)
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', calibrated_svm)
    ])

    # 2. Pipeline Random Forest con StandardScaler
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', rf_clf)
    ])

    # 3. Ensamble Híbrido SVM-RF (Soft Voting)
    hybrid_clf = VotingClassifier(
        estimators=[('svm', svm_pipe), ('rf', rf_pipe)],
        voting='soft',
        n_jobs=-1
    )

    return svm_pipe, rf_pipe, hybrid_clf

def run():
    print("=================================================================")
    print("  ETAPA 4: ENTRENAMIENTO DE MODELOS BASE E HÍBRIDO (SVM + RF)")
    print("=================================================================")
    
    opt_dir, split_dir = setup_directories()
    # Incluimos las tres modalidades unimodales y la fusión multimodal
    modalities = ['face', 'fingerprint', 'iris', 'multimodal']
    count_models = 0
    
    for mod in modalities:
        x_path = opt_dir / f'X_{mod}_opt.npy'
        y_path = opt_dir / f'y_{mod}.npy'
        
        if not x_path.exists() or not y_path.exists():
            continue
            
        print(f"\n--- Entrenando arquitecturas para: {mod.upper()} ---")
        X = np.load(str(x_path))
        y = np.load(str(y_path))
        
        # 1. Partición estratificada (70% Train, 30% Test ciego)
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        use_stratify = y if min_count >= 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=use_stratify, random_state=42
        )
        
        print(f"[+] Conjunto Train: {X_train.shape[0]} muestras | Test (ciego): {X_test.shape[0]} muestras.")
        
        # Guardamos las porciones de Test para la evaluación ciega en la Etapa 5
        np.save(str(split_dir / f'X_test_{mod}.npy'), X_test)
        np.save(str(split_dir / f'y_test_{mod}.npy'), y_test)
        
        # 2. Instanciación y Entrenamiento
        svm_pipe, rf_pipe, hybrid_clf = build_models()
        
        print("[+] Entrenando modelo SVM (con StandardScaler y calibración)...")
        svm_pipe.fit(X_train, y_train)
        
        print("[+] Entrenando modelo Random Forest (100 árboles)...")
        rf_pipe.fit(X_train, y_train)
        
        print("[+] Entrenando modelo Híbrido SVM-RF (Soft Voting)...")
        hybrid_clf.fit(X_train, y_train)
        
        # 3. Serialización de los tres modelos para comparativa científica completa
        joblib.dump(svm_pipe, str(split_dir / f'svm_model_{mod}.joblib'))
        joblib.dump(rf_pipe, str(split_dir / f'rf_model_{mod}.joblib'))
        joblib.dump(hybrid_clf, str(split_dir / f'hybrid_model_{mod}.joblib'))
        
        print(f"[+] Modelos guardados para {mod.upper()}:")
        print(f"    - svm_model_{mod}.joblib")
        print(f"    - rf_model_{mod}.joblib")
        print(f"    - hybrid_model_{mod}.joblib")
        count_models += 1
        
    if count_models > 0:
        print("\n[+] ¡Éxito! Todos los modelos (unimodales y multimodal) han sido entrenados y compilados.")
    else:
        print("\n[-] No se encontraron datos optimizados. Por favor corre la Etapa 3 primero.")

if __name__ == "__main__":
    run()
