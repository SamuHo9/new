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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, recall_score, f1_score, roc_auc_score

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

def extract_features_labels(df):
    meta_cols = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType', 'Group_Name', 'Group_Label', 'Unnamed: 0']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    features_3d_raw = df[feature_cols].values
    
    num_points = features_3d_raw.shape[1] // 3
    features_3d = features_3d_raw.reshape(-1, num_points, 3)
    
    mean_val = np.mean(features_3d, axis=1, keepdims=True)
    std_val = np.std(features_3d, axis=1, keepdims=True) + 1e-8
    features_3d = (features_3d - mean_val) / std_val
    
    X = np.transpose(features_3d, (0, 2, 1))
    
    if 'BinaryClass' in df.columns:
        y = df['BinaryClass'].values
    elif 'Group_Label' in df.columns:
        y = df['Group_Label'].values
    elif 'Class' in df.columns:
        y = df['Class'].values
    else:
        raise ValueError("No binary target label column found")
        
    return X, y

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k*k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 512)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = (x + iden).view(-1, self.k, self.k)
        return x

class PointNetDual(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetDual, self).__init__()
        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(1024, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        trans3 = self.input_transform(x)
        x = torch.bmm(trans3, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x_max = torch.max(x, 2)[0]
        x_mean = torch.mean(x, 2)
        global_feat = torch.cat([x_max, x_mean], dim=1)
        
        x = F.relu(self.bn4(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

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

    print("Loading point clouds...")
    train_df = pd.read_csv('../Ds005602_Right_train_xyz_coords.csv')
    test_df = pd.read_csv('../Ds005602_Right_test_xyz_coords.csv')

    X_train, y_train = extract_features_labels(train_df)
    X_test, y_test = extract_features_labels(test_df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    c0, c1 = np.sum(y_train == 0), np.sum(y_train == 1)
    weights = torch.tensor([len(y_train)/(2.0*c0), len(y_train)/(2.0*c1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    n_splits = min(10, min(c0, c1))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_models = []
    cv_accuracies = []

    batch_size = min(32, max(8, len(X_train) // 4))
    print(f"Running PointNet Dual Pooling {n_splits}-Fold CV with Checkpointing (Batch Size: {batch_size})...")

    patience = 25
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_t.cpu(), y_train_t.cpu())):
        X_tr, y_tr = X_train_t[train_idx], y_train_t[train_idx]
        X_val, y_val = X_train_t[val_idx], y_train_t[val_idx]
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        
        model = PointNetDual(num_classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        
        best_loss = float('inf')
        best_weights = copy.deepcopy(model.state_dict())
        best_ep = 1
        no_improve = 0
        actual_ep = 1
        
        for epoch in range(1, 81):
            actual_ep = epoch
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
                    total += batch_x.size(0)
            val_loss /= total
            
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_ep = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
                
        best_model = PointNetDual(num_classes=2).to(device)
        best_model.load_state_dict(best_weights)
        best_model.eval()
        with torch.no_grad():
            preds = torch.argmax(best_model(X_val), dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val.cpu().numpy(), preds)
        
        cv_accuracies.append(val_acc)
        fold_models.append(best_model)
        if actual_ep < 80:
            print(f"  Fold {fold+1:2d} / {n_splits} Best Val Acc: {val_acc:.4f} (Best Checkpoint: Epoch {best_ep}, Early stopped: Epoch {actual_ep})")
        else:
            print(f"  Fold {fold+1:2d} / {n_splits} Best Val Acc: {val_acc:.4f} (Best Checkpoint: Epoch {best_ep} / 80)")

    mean_cv = np.mean(cv_accuracies)
    print(f"Mean CV Accuracy: {mean_cv:.4f} (+/- {np.std(cv_accuracies):.4f})")
    
    print("")
    print("Evaluating on Test Set...")
    probs_list = []
    for m in fold_models:
        m.eval()
        with torch.no_grad():
            probs = F.softmax(m(X_test_t), dim=1)[:, 1].cpu().numpy()
            probs_list.append(probs)

    y_prob = np.mean(probs_list, axis=0)
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
    plt.title(f'PointNet Confusion Matrix (Acc: {test_acc:.2%})')
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
