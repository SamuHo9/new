import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

np.random.seed(42)

def run_pipeline():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'logs'))
        os.makedirs(logs_dir, exist_ok=True)
        side = os.path.basename(os.path.abspath(os.path.join(script_dir, '..')))
        m_name = os.path.basename(script_dir)
        s_stem = os.path.splitext(os.path.basename(__file__))[0]
        log_file = os.path.join(logs_dir, f"{side}_{m_name}_{s_stem}.log")
        sys.stdout = Logger(log_file)
    except Exception as e:
        pass

    print("Loading data...")
    train_df = pd.read_csv('../ALL_Left_train_coef_features.csv')
    test_df = pd.read_csv('../ALL_Left_test_coef_features.csv')

    meta_cols = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType', 'Group_Name', 'Group_Label', 'Unnamed: 0']
    
    train_drop = [c for c in meta_cols if c in train_df.columns]
    X_train = train_df.drop(columns=train_drop)
    y_train = train_df['BinaryClass'].values if 'BinaryClass' in train_df.columns else train_df['Group_Label'].values

    test_drop = [c for c in meta_cols if c in test_df.columns]
    X_test = test_df.drop(columns=test_drop)
    y_test = test_df['BinaryClass'].values if 'BinaryClass' in test_df.columns else test_df['Group_Label'].values
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    c0, c1 = np.sum(y_train == 0), np.sum(y_train == 1)
    n_splits = min(10, min(c0, c1))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    components_to_try = [2, 5, 10, 15, 20, 30, 40, 50]
    max_comp = min(int(X_train.shape[0] * 0.8), X_train.shape[1])
    components_to_try = [c for c in components_to_try if c <= max_comp]

    print("Evaluating PLS-DA components...")
    pls_cv_scores = []
    for n_comp in components_to_try:
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train.iloc[train_idx])
            X_val_sc = scaler.transform(X_train.iloc[val_idx])
            
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr_sc, y_train[train_idx])
            y_pred_val = pls.predict(X_val_sc)
            scores.append(accuracy_score(y_train[val_idx], (y_pred_val > 0.5).astype(int).flatten()))
        mean_score = np.mean(scores)
        pls_cv_scores.append(mean_score)
        print(f"PLS components: {n_comp}, CV Accuracy: {mean_score:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(components_to_try, pls_cv_scores, marker='o', color='royalblue')
    plt.title('PLS Components vs CV Accuracy')
    plt.xlabel('Number of Components')
    plt.ylabel('CV Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/pls_components_comparison.png')
    plt.close()
    print("Saved PLS components comparison graph to 'plots/pls_components_comparison.png'")

    best_n_comp = components_to_try[np.argmax(pls_cv_scores)]
    print(f"-> Selected Best number of PLS components: {best_n_comp} with CV accuracy: {np.max(pls_cv_scores):.4f}")

    print(f"Fitting final PLS model with {best_n_comp} components...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    pls_final = PLSRegression(n_components=best_n_comp)
    pls_final.fit(X_train_sc, y_train)
    X_train_pls = pls_final.transform(X_train_sc)
    X_test_pls = pls_final.transform(X_test_sc)

    print(f"Training SVM Model with {n_splits}-Fold CV...")
    svm_cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_pls, y_train)):
        m = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42)
        m.fit(X_train_pls[train_idx], y_train[train_idx])
        val_acc = accuracy_score(y_train[val_idx], m.predict(X_train_pls[val_idx]))
        svm_cv_scores.append(val_acc)
        print(f"  Fold {fold+1:2d} / {n_splits} Validation Acc: {val_acc:.4f}")
    
    cv_acc = np.mean(svm_cv_scores)
    print(f"SVM {n_splits}-Fold CV Accuracy (Strict without leakage): {cv_acc:.4f} (+/- {np.std(svm_cv_scores):.4f})")

    svm = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_pls, y_train)
    y_pred = svm.predict(X_test_pls)
    y_prob = svm.predict_proba(X_test_pls)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

    print("")
    print("Evaluating on Test Set...")
    print(f"*** Final Test Accuracy: {test_acc:.4f} ***")
    print("")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=2))
    print(f"Sensitivity (TLE Recall): {sens:.4f} | Specificity (Healthy Recall): {spec:.4f} | F1-Macro: {f1_m:.4f} | ROC-AUC: {auc_score:.4f}")

    np.savez('results/test_predictions.npz', y_test=y_test, y_pred=y_pred, y_prob=y_prob)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'TLE'], yticklabels=['Healthy', 'TLE'])
    plt.title(f'SVM Confusion Matrix (Acc: {test_acc:.2%})')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    print("Saved Confusion Matrix to 'plots/confusion_matrix.png'")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png')
    plt.close()
    print("Saved ROC curve to 'plots/roc_curve.png'")
    print("")
    print("Pipeline finished successfully! All files are in the 'plots' directory.")

if __name__ == "__main__":
    run_pipeline()
