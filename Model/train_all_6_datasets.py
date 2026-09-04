import os
import sys
import subprocess
import time
import re
import csv

base_model_dir = os.path.abspath(os.path.dirname(__file__))

datasets = [
    ("Ds005602", ["left", "right"]),
    ("Ds004469", ["left", "right"]),
    ("All_Augment_tain", ["left", "right"]),
]

models = [
    ("SVM", "train_svm_pls.py"),
    ("MLP", "train_mlp_pls.py"),
    ("ResNet (Standard)", "train_resnet_pls.py", "ResNet"),
    ("ResNet (AutoEncoder)", "train_resnet_ae_pls.py", "ResNet"),
    ("MobileNet", "train_mobilenet_pls.py"),
    ("SqueezeNet", "train_squeezenet_pls.py"),
    ("PointNet", "train_pointnet.py"),
]

results_summary = []

total_start = time.time()
print("=" * 70)
print("STARTING 10-FOLD CV BATCH TRAINING PIPELINE ACROSS ALL DATASETS")
print("=" * 70)

for ds_name, sides in datasets:
    ds_dir = os.path.join(base_model_dir, ds_name)
    logs_dir = os.path.join(ds_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    for side in sides:
        side_dir = os.path.join(ds_dir, side)
        if not os.path.isdir(side_dir):
            print(f"[SKIP] Directory not found: {side_dir}")
            continue
            
        dataset_label = f"{ds_name} ({side.upper()})"
        print(f"\n========================================================")
        print(f"  DATASET: {dataset_label}")
        print(f"========================================================")
        
        for item in models:
            model_name = item[0]
            script_file = item[1]
            subfolder = item[2] if len(item) > 2 else model_name
            
            script_path = os.path.join(side_dir, subfolder, script_file)
            if not os.path.isfile(script_path):
                print(f"  [MISSING SCRIPT]: {script_path}")
                continue
                
            log_name = f"{side}_{subfolder}_{os.path.splitext(script_file)[0]}.log"
            log_file = os.path.join(logs_dir, log_name)
            
            print(f"  >> Training {model_name} (10-Fold CV Checkpoint Ensemble)...")
            start_t = time.time()
            
            proc = subprocess.run(
                [sys.executable, script_file],
                cwd=os.path.dirname(script_path),
                capture_output=True,
                text=True
            )
            elapsed = time.time() - start_t
            
            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write(proc.stdout)
                if proc.stderr and proc.returncode != 0:
                    lf.write(f"\n=== STDERR ===\n{proc.stderr}\n")
                
            test_acc = "N/A"
            cv_acc = "N/A"
            sens = "N/A"
            spec = "N/A"
            f1_macro = "N/A"
            roc_auc = "N/A"
            
            m_test = re.search(r"\*\*\* Final Test Accuracy:\s*([0-9.]+)\s*\*\*\*", proc.stdout) or re.search(r"Final Test Accuracy:\s*([0-9.]+)", proc.stdout)
            if m_test:
                test_acc = f"{float(m_test.group(1))*100:.2f}%"
                
            m_cv = re.search(r"(?:Mean CV Accuracy|CV Accuracy \(Strict without leakage\)|with CV accuracy):\s*([0-9.]+)", proc.stdout)
            if m_cv:
                cv_acc = f"{float(m_cv.group(1))*100:.2f}%"

            m_metrics = re.search(r"Sensitivity \(TLE Recall\):\s*([0-9.]+)\s*\|\s*Specificity \(Healthy Recall\):\s*([0-9.]+)\s*\|\s*F1-Macro:\s*([0-9.]+)\s*\|\s*ROC-AUC:\s*([0-9.]+)", proc.stdout)
            if m_metrics:
                sens = f"{float(m_metrics.group(1))*100:.2f}%"
                spec = f"{float(m_metrics.group(2))*100:.2f}%"
                f1_macro = f"{float(m_metrics.group(3))*100:.2f}%"
                roc_auc = f"{float(m_metrics.group(4)):.4f}"
                
            status = "OK" if proc.returncode == 0 else "FAIL"
            print(f"     Status: {status} | CV: {cv_acc} | Test: {test_acc} | Sens: {sens} | Spec: {spec} | F1: {f1_macro} | AUC: {roc_auc} | Time: {elapsed:.1f}s", flush=True)
            
            results_summary.append({
                "Dataset": ds_name,
                "Side": side.capitalize(),
                "Model": model_name,
                "Status": status,
                "CV_Acc": cv_acc,
                "Test_Acc": test_acc,
                "Sensitivity": sens,
                "Specificity": spec,
                "F1_Macro": f1_macro,
                "ROC_AUC": roc_auc,
                "Time": f"{elapsed:.1f}s"
            })

total_time = time.time() - total_start
print("\n" + "=" * 70, flush=True)
print(f"ALL 10-FOLD CV TRAINING COMPLETED IN {total_time/60:.2f} MINUTES", flush=True)
print("=" * 70, flush=True)

# Print Summary Table
print("\n" + "=" * 105, flush=True)
print("10-FOLD CV TRAINING BENCHMARK SUMMARY", flush=True)
print("=" * 105, flush=True)
print(f"{'Dataset':<16} {'Side':<7} {'Model':<22} {'CV Acc':<10} {'Test Acc':<10} {'Sens':<8} {'Spec':<8} {'F1-M':<8} {'AUC':<8}", flush=True)
print("-" * 105, flush=True)
for r in results_summary:
    print(f"{r['Dataset']:<16} {r['Side']:<7} {r['Model']:<22} {r['CV_Acc']:<10} {r['Test_Acc']:<10} {r['Sensitivity']:<8} {r['Specificity']:<8} {r['F1_Macro']:<8} {r['ROC_AUC']:<8}", flush=True)

summary_csv = os.path.join(base_model_dir, "all_models_training_summary.csv")
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["Dataset", "Side", "Model", "Status", "CV_Acc", "Test_Acc", "Sensitivity", "Specificity", "F1_Macro", "ROC_AUC", "Time"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results_summary)

print(f"\nSaved 10-Fold training benchmark table to: {summary_csv}", flush=True)
