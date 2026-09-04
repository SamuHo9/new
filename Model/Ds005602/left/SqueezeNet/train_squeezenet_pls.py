import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cross_decomposition import PLSRegression
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
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class Fire1D(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire1D, self).__init__()
        self.squeeze = nn.Conv1d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv1d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv1d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(SqueezeNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Fire1D(64, 16, 64, 64),
            Fire1D(128, 16, 64, 64),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
            Fire1D(128, 32, 128, 128),
            Fire1D(256, 32, 128, 128),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Conv1d(256, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)


def train_with_checkpoint(model, X_tr, y_tr, X_val, y_val, epochs=80, batch_size=32, lr=0.001, device='cpu', pos_weight=1.0, patience=25):
    pos_w_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 1
    no_improve = 0
    actual_epochs = 1
    
    for epoch in range(1, epochs + 1):
        actual_epochs = epoch
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item() * bx.size(0)
                total += bx.size(0)
        val_loss /= total
        
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
            
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
        preds = (torch.sigmoid(model(X_val_t)).cpu().numpy().flatten() > 0.5).astype(int)
        val_acc = accuracy_score(y_val, preds)
        
    return model, val_acc, best_epoch, actual_epochs

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_df = pd.read_csv('../Ds005602_Left_train_coef_features.csv')
    test_df = pd.read_csv('../Ds005602_Left_test_coef_features.csv')

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
    pos_weight = float(c0) / float(c1) if c1 > 0 else 1.0

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

    batch_size = min(32, max(8, len(X_train) // 8))
    print(f"Running SqueezeNet1D {n_splits}-Fold CV with Checkpointing (Batch Size: {batch_size})...")

    fold_models = []
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train.iloc[train_idx])
        X_val_sc = scaler.transform(X_train.iloc[val_idx])
        
        pls = PLSRegression(n_components=best_n_comp)
        pls.fit(X_tr_sc, y_train[train_idx])
        X_tr_pls = pls.transform(X_tr_sc)
        X_val_pls = pls.transform(X_val_sc)
        
        model = SqueezeNet1D().to(device)
        model, val_acc, best_ep, actual_ep = train_with_checkpoint(
            model, X_tr_pls, y_train[train_idx], X_val_pls, y_train[val_idx], 
            epochs=80, batch_size=batch_size, device=device, pos_weight=pos_weight, patience=25
        )
        cv_scores.append(val_acc)
        fold_models.append((scaler, pls, model))
        if actual_ep < 80:
            print(f"  Fold {fold+1:2d} / {n_splits} Best Val Acc: {val_acc:.4f} (Best Checkpoint: Epoch {best_ep}, Early stopped: Epoch {actual_ep})")
        else:
            print(f"  Fold {fold+1:2d} / {n_splits} Best Val Acc: {val_acc:.4f} (Best Checkpoint: Epoch {best_ep} / 80)")

    mean_cv = np.mean(cv_scores)
    print(f"Mean CV Accuracy: {mean_cv:.4f} (+/- {np.std(cv_scores):.4f})")
    
    print("")
    print("Evaluating on Test Set...")
    test_probs_folds = []
    for scaler, pls, model in fold_models:
        model.eval()
        with torch.no_grad():
            X_test_sc = scaler.transform(X_test)
            X_test_pls = pls.transform(X_test_sc)
            X_test_t = torch.tensor(X_test_pls, dtype=torch.float32).unsqueeze(1).to(device)
            probs = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()
            test_probs_folds.append(probs)

    y_prob = np.mean(test_probs_folds, axis=0)
    y_pred = (y_prob > 0.5).astype(int)

    test_acc = accuracy_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

    print(f"*** Final Test Accuracy: {test_acc:.4f} ***")
    print("")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=2))
    print(f"Sensitivity (TLE Recall): {sens:.4f} | Specificity (Healthy Recall): {spec:.4f} | F1-Macro: {f1_m:.4f} | ROC-AUC: {auc_score:.4f}")

    np.savez('results/test_predictions.npz', y_test=y_test, y_pred=y_pred, y_prob=y_prob)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'TLE'], yticklabels=['Healthy', 'TLE'])
    plt.title(f'SqueezeNet1D Confusion Matrix (Acc: {test_acc:.2%})')
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
