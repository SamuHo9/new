import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

np.random.seed(42)

def compute_bootstrap_stats(y_true, y_prob, n_boot=1000):
    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob.flatten()
            
    y_pred = (y_prob > 0.5).astype(int)
    n = len(y_true)
    
    # Generate all bootstrap sample indices at once: shape (n_boot, n)
    boot_indices = np.random.randint(0, n, size=(n_boot, n))
    yt_all = y_true[boot_indices]
    yp_all = y_pred[boot_indices]
    ypr_all = y_prob[boot_indices]
    
    # Accuracy per bootstrap
    boot_acc = np.mean(yt_all == yp_all, axis=1)
    
    # Sensitivity (pos = 1)
    pos_mask = (yt_all == 1)
    pos_correct = (yp_all == 1) & pos_mask
    pos_total = np.sum(pos_mask, axis=1)
    boot_sens = np.divide(np.sum(pos_correct, axis=1), pos_total, out=np.zeros(n_boot), where=(pos_total > 0))
    
    # Specificity (pos = 0)
    neg_mask = (yt_all == 0)
    neg_correct = (yp_all == 0) & neg_mask
    neg_total = np.sum(neg_mask, axis=1)
    boot_spec = np.divide(np.sum(neg_correct, axis=1), neg_total, out=np.zeros(n_boot), where=(neg_total > 0))
    
    # F1 Macro
    tp1 = np.sum((yp_all == 1) & (yt_all == 1), axis=1)
    fp1 = np.sum((yp_all == 1) & (yt_all == 0), axis=1)
    fn1 = np.sum((yp_all == 0) & (yt_all == 1), axis=1)
    denom1 = 2 * tp1 + fp1 + fn1
    f1_class1 = np.divide(2 * tp1, denom1, out=np.zeros(n_boot), where=(denom1 > 0))
    
    tp0 = np.sum((yp_all == 0) & (yt_all == 0), axis=1)
    fp0 = np.sum((yp_all == 0) & (yt_all == 1), axis=1)
    fn0 = np.sum((yp_all == 1) & (yt_all == 0), axis=1)
    denom0 = 2 * tp0 + fp0 + fn0
    f1_class0 = np.divide(2 * tp0, denom0, out=np.zeros(n_boot), where=(denom0 > 0))
    
    boot_f1 = 0.5 * (f1_class1 + f1_class0)
    
    # AUC using Mann-Whitney U rank approximation
    boot_auc = []
    for i in range(n_boot):
        yt = yt_all[i]
        ypr = ypr_all[i]
        n_pos = np.sum(yt == 1)
        n_neg = np.sum(yt == 0)
        if n_pos > 0 and n_neg > 0:
            rank = np.argsort(np.argsort(ypr)) + 1
            u = np.sum(rank[yt == 1]) - n_pos * (n_pos + 1) / 2.0
            boot_auc.append(u / (n_pos * n_neg))
        else:
            boot_auc.append(0.5)
            
    acc_m, acc_std = np.mean(boot_acc)*100, np.std(boot_acc)*100
    acc_ci = f"[{np.percentile(boot_acc, 2.5)*100:.2f}%, {np.percentile(boot_acc, 97.5)*100:.2f}%]"
    
    sens_m = np.mean(boot_sens)*100 if len(boot_sens) > 0 else 0.0
    sens_ci = f"[{np.percentile(boot_sens, 2.5)*100:.2f}%, {np.percentile(boot_sens, 97.5)*100:.2f}%]" if len(boot_sens) > 0 else "[0.00%, 0.00%]"
    
    spec_m = np.mean(boot_spec)*100 if len(boot_spec) > 0 else 0.0
    spec_ci = f"[{np.percentile(boot_spec, 2.5)*100:.2f}%, {np.percentile(boot_spec, 97.5)*100:.2f}%]" if len(boot_spec) > 0 else "[0.00%, 0.00%]"
    
    f1_m = np.mean(boot_f1)*100
    f1_ci = f"[{np.percentile(boot_f1, 2.5)*100:.2f}%, {np.percentile(boot_f1, 97.5)*100:.2f}%]"
    
    auc_m = np.mean(boot_auc) if len(boot_auc) > 0 else 0.5
    auc_ci = f"[{np.percentile(boot_auc, 2.5):.4f}, {np.percentile(boot_auc, 97.5):.4f}]" if len(boot_auc) > 0 else "[0.5000, 0.5000]"
    
    return {
        "Mean_Acc": f"{acc_m:.2f}%",
        "Acc_SD": f"+/-{acc_std:.2f}%",
        "Acc_95CI": acc_ci,
        "Mean_Sens": f"{sens_m:.2f}%",
        "Sens_95CI": sens_ci,
        "Mean_Spec": f"{spec_m:.2f}%",
        "Spec_95CI": spec_ci,
        "Mean_F1": f"{f1_m:.2f}%",
        "F1_95CI": f1_ci,
        "Mean_AUC": f"{auc_m:.4f}",
        "AUC_95CI": auc_ci
    }

def run_all():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    datasets = [
        "Ds005602", 
        "All_Augment_tain", 
        "Ds004469",
        "Ds004469Train_Ds005602test",
        "Ds005602Train_Ds004469test"
    ]
    sides = ["left", "right"]
    
    models_config = [
        ("SVM", "SVM", "test_predictions.npz"),
        ("MLP", "MLP", "test_predictions.npz"),
        ("ResNet (Standard)", "ResNet", "test_predictions.npz"),
        ("ResNet (AutoEncoder)", "ResNet", "test_predictions_ae.npz"),
        ("MobileNet", "MobileNet", "test_predictions.npz"),
        ("SqueezeNet", "SqueezeNet", "test_predictions.npz"),
        ("PointNet", "PointNet", "test_predictions.npz"),
    ]
    
    results = []
    
    for ds in datasets:
        print(f"\n========================================================")
        print(f"  BOOTSTRAP 1,000 RUNS: DATASET = {ds}")
        print(f"========================================================")
        for side in sides:
            side_label = side.upper()
            print(f"\n--- Side: {side_label} ---")
            for m_name, subfolder, npz_name in models_config:
                npz_path = os.path.join(base_dir, ds, side, subfolder, "results", npz_name)
                if not os.path.exists(npz_path):
                    continue
                
                data = np.load(npz_path)
                y_true = data['y_test'] if 'y_test' in data else data['y_true']
                y_prob = data['y_prob']
                
                stats = compute_bootstrap_stats(y_true, y_prob, n_boot=1000)
                stats.update({
                    "Dataset": ds,
                    "Side": side.capitalize(),
                    "Model": m_name,
                    "Test_N": len(y_true),
                    "Pos_N": int(np.sum(y_true == 1)),
                    "Neg_N": int(np.sum(y_true == 0))
                })
                results.append(stats)
                print(f"  [{m_name:<20}] Mean Acc: {stats['Mean_Acc']:<7} {stats['Acc_SD']:<9} (95% CI: {stats['Acc_95CI']:<20}) | Sens: {stats['Mean_Sens']:<7} | Spec: {stats['Mean_Spec']:<7} | F1: {stats['Mean_F1']:<7} | AUC: {stats['Mean_AUC']}")

    df_res = pd.DataFrame(results)
    
    # Save CSVs
    out_csv = os.path.join(base_dir, "bootstrap_1000_summary.csv")
    df_res.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"\nSaved Bootstrap summary to: {out_csv}")
    
    # Also save to Model_Evaluation_and_Benchmarks
    bench_csv = os.path.abspath(os.path.join(base_dir, "..", "Model_Evaluation_and_Benchmarks", "results_csv", "bootstrap_1000_summary.csv"))
    os.makedirs(os.path.dirname(bench_csv), exist_ok=True)
    df_res.to_csv(bench_csv, index=False, encoding='utf-8')
    print(f"Saved to dedicated benchmark folder: {bench_csv}")

if __name__ == "__main__":
    run_all()
