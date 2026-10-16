import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib
import warnings

# Ignoramos warnings visuales de matplotlib y sklearn para limpieza de consola
warnings.filterwarnings('ignore')

def setup_directories():
    base_dir = Path(__file__).resolve().parent.parent
    split_dir = base_dir / 'data' / '5_splits'
    
    # Creamos una carpeta nueva para guardar el material del artículo (los gráficos)
    res_dir = base_dir / 'results'
    res_dir.mkdir(parents=True, exist_ok=True)
    return split_dir, res_dir

def evaluate_model(modality, split_dir, res_dir):
    """
    Carga el modelo híbrido previamente entrenado y lo somete al examen final
    con el 30% de datos ciegos. Genera las métricas y los gráficos.
    """
    model_path = split_dir / f'hybrid_model_{modality}.joblib'
    x_test_path = split_dir / f'X_test_{modality}.npy'
    y_test_path = split_dir / f'y_test_{modality}.npy'
    
    if not (model_path.exists() and x_test_path.exists() and y_test_path.exists()):
        return False
        
    print(f"\n--- Evaluando Biometría: {modality.upper()} ---")
    
    # 1. Cargar datos e IA del disco duro a la memoria RAM
    model = joblib.load(str(model_path))
    X_test = np.load(str(x_test_path))
    y_test = np.load(str(y_test_path))
    
    # 2. El Examen Final (Predicciones ciegas)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) # Probabilidades necesarias para la curva ROC
    
    # 3. Métricas Científicas
    acc = accuracy_score(y_test, y_pred)
    # Usamos average='weighted' por si hay desbalance de clases (Ej. en el dataset de rostros)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"[>] Exactitud Global (Accuracy): {acc*100:.2f}%")
    print(f"[>] Precisión: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1-Score: {f1*100:.2f}%")
    
    # 4. Generación Gráfica: Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {modality.capitalize()}')
    plt.xlabel('Predicción del SVM-RF')
    plt.ylabel('Etiqueta Real')
    cm_path = res_dir / f'confusion_matrix_{modality}.png'
    plt.savefig(str(cm_path), bbox_inches='tight')
    plt.close()
    
    # 5. Generación Gráfica: Curva ROC
    classes = np.unique(y_test)
    n_classes = len(classes)
    
    plt.figure(figsize=(7, 6))
    if n_classes == 2:
        # Lógica para datasets Binarios (Cara: Real vs Falsa / Huella: Real vs Alterada)
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        # Lógica para datasets Multiclase (Iris: 46 personas distintas)
        y_test_bin = label_binarize(y_test, classes=classes)
        for i in range(n_classes):
            # Limitamos a plotear solo las primeras 5 clases para no saturar el gráfico
            if i >= 5: break 
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Clase {classes[i]} (AUC = {roc_auc:.2f})')
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (False Acceptance)')
    plt.ylabel('Tasa de Verdaderos Positivos (True Acceptance)')
    plt.title(f'Curva ROC - {modality.capitalize()}')
    plt.legend(loc="lower right")
    
    roc_path = res_dir / f'roc_curve_{modality}.png'
    plt.savefig(str(roc_path), bbox_inches='tight')
    plt.close()
    
    print(f"[+] Gráficos científicos generados en: {res_dir.name}/")
    return True

def run():
    print("======================================================")
    print("  ETAPA 5: EVALUACIÓN Y GRÁFICOS CIENTÍFICOS")
    print("======================================================")
    split_dir, res_dir = setup_directories()
    
    modalities = ['face', 'fingerprint', 'iris']
    evaluados = 0
    
    for mod in modalities:
        if evaluate_model(mod, split_dir, res_dir):
            evaluados += 1
            
    if evaluados == 0:
        print("\n[-] Error: No se encontraron modelos o datos de prueba. Corre la Etapa 4 primero.")

if __name__ == "__main__":
    run()
