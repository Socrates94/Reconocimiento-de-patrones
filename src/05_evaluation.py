import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin pantalla
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)
from sklearn.preprocessing import label_binarize
import joblib
import warnings

warnings.filterwarnings('ignore')

def setup_directories():
    base_dir = Path(__file__).resolve().parent.parent
    split_dir = base_dir / 'data' / '5_splits'
    res_dir = base_dir / 'results'
    res_dir.mkdir(parents=True, exist_ok=True)
    return split_dir, res_dir

def compute_metrics(y_true, y_pred, y_prob):
    """Calcula el conjunto completo de métricas científicas."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Pérdida (Cross-Entropy / Log Loss)
    try:
        loss = log_loss(y_true, y_prob)
    except Exception:
        loss = float('nan')
        
    # Especificidad y AUC (para binario)
    classes = np.unique(y_true)
    specificity = float('nan')
    roc_auc = float('nan')
    
    if len(classes) == 2:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
        except Exception:
            pass
    elif len(classes) > 2:
        try:
            y_bin = label_binarize(y_true, classes=classes)
            # Promedio micro/macro de AUC
            aucs = []
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                aucs.append(auc(fpr, tpr))
            roc_auc = np.mean(aucs)
        except Exception:
            pass
            
    return {
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'loss': loss, 'spec': specificity, 'auc': roc_auc
    }

def evaluate_modality(modality, split_dir, res_dir):
    """
    Evalúa SVM, Random Forest y el modelo Híbrido sobre los datos ciegos de test (30%).
    """
    x_test_path = split_dir / f'X_test_{modality}.npy'
    y_test_path = split_dir / f'y_test_{modality}.npy'
    
    if not (x_test_path.exists() and y_test_path.exists()):
        return None
        
    X_test = np.load(str(x_test_path))
    y_test = np.load(str(y_test_path))
    
    models = {}
    for name in ['svm', 'rf', 'hybrid']:
        m_path = split_dir / f'{name}_model_{modality}.joblib'
        if m_path.exists():
            models[name] = joblib.load(str(m_path))
            
    if 'hybrid' not in models:
        return None
        
    print(f"\n=======================================================")
    print(f"  EVALUACIÓN CIENTÍFICA: {modality.upper()} (N_test={len(y_test)})")
    print(f"=======================================================")
    
    results = {}
    preds_dict = {}
    probs_dict = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        results[name] = metrics
        preds_dict[name] = y_pred
        probs_dict[name] = y_prob
        
        loss_str = f"{metrics['loss']:.2f}" if not np.isnan(metrics['loss']) else "N/A"
        auc_str = f"{metrics['auc']:.2f}" if not np.isnan(metrics['auc']) else "N/A"
        spec_str = f"{metrics['spec']*100:.2f}%" if not np.isnan(metrics['spec']) else "N/A"
        
        print(f"[{name.upper():6s}] Acc: {metrics['acc']*100:6.2f}% | Prec: {metrics['prec']*100:6.2f}% | Rec: {metrics['rec']*100:6.2f}% | F1: {metrics['f1']*100:6.2f}% | AUC: {auc_str} | Loss: {loss_str} | Spec: {spec_str}")
        
    # 1. Matriz de Confusión del Modelo Híbrido
    cm = confusion_matrix(y_test, preds_dict['hybrid'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión ({modality.capitalize()}) - Híbrido SVM-RF')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    cm_path = res_dir / f'confusion_matrix_{modality}.png'
    plt.savefig(str(cm_path), bbox_inches='tight', dpi=150)
    plt.close()
    
    # 2. Curva ROC Comparativa (SVM vs RF vs Híbrido)
    classes = np.unique(y_test)
    if len(classes) == 2:
        plt.figure(figsize=(7, 6))
        colors = {'svm': 'blue', 'rf': 'green', 'hybrid': 'darkorange'}
        labels = {'svm': 'SVM (Kernel Lineal)', 'rf': 'Random Forest', 'hybrid': 'Híbrido SVM-RF (Propuesto)'}
        
        for name in ['svm', 'rf', 'hybrid']:
            if name in probs_dict:
                fpr, tpr, _ = roc_curve(y_test, probs_dict[name][:, 1])
                score_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors.get(name, 'black'), lw=2,
                         label=f"{labels.get(name, name)} (AUC = {score_auc:.2f})")
                
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title(f'Curvas ROC - {modality.capitalize()}')
        plt.legend(loc="lower right")
        roc_path = res_dir / f'roc_curve_{modality}.png'
        plt.savefig(str(roc_path), bbox_inches='tight', dpi=150)
        plt.close()
        
    return results

def run():
    print("=================================================================")
    print("  ETAPA 5: EVALUACIÓN COMPARATIVA Y GENERACIÓN DE RESULTADOS")
    print("=================================================================")
    split_dir, res_dir = setup_directories()
    
    modalities = ['face', 'fingerprint', 'iris', 'multimodal']
    all_results = {}
    
    for mod in modalities:
        res = evaluate_modality(mod, split_dir, res_dir)
        if res:
            all_results[mod] = res
            
    if not all_results:
        print("\n[-] Error: No se encontraron modelos o datos de test. Ejecuta la Etapa 4 primero.")
        return

    # Tabla resumen final en consola
    print("\n\n" + "=" * 80)
    print("                     TABLA COMPARATIVA DE RESULTADOS")
    print("=" * 80)
    header = f"{'Biometría':<14} | {'Modelo':<12} | {'Exactitud':<10} | {'Precisión':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC':<6}"
    print(header)
    print("-" * 80)
    
    for mod, models_res in all_results.items():
        for m_name in ['svm', 'rf', 'hybrid']:
            if m_name in models_res:
                r = models_res[m_name]
                auc_s = f"{r['auc']:.2f}" if not np.isnan(r['auc']) else "N/A"
                print(f"{mod.upper():<14} | {m_name.upper():<12} | {r['acc']*100:8.2f}% | {r['prec']*100:8.2f}% | {r['rec']*100:8.2f}% | {r['f1']*100:8.2f}% | {auc_s:<6}")
        print("-" * 80)
    print(f"\n[+] Gráficos científicos guardados en la carpeta: {res_dir.resolve()}/")

if __name__ == "__main__":
    run()
