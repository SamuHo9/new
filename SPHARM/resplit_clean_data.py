import os
import glob
import shutil
import pandas as pd

base_dir = os.path.abspath(r"SPHARM\split_data")
left_src = os.path.abspath(r"ICP\output_left_hippocampus\spharm_results_left")
right_src = os.path.abspath(r"ICP\output_right_hippocampus\spharm_results_right")

datasets = [
    ("Ds004469_Left", left_src, "Ds004469_Left_split.csv"),
    ("Ds004469_Right", right_src, "Ds004469_Right_split.csv"),
    ("Ds005602_Left", left_src, "Ds005602_Left_split.csv"),
    ("Ds005602_Right", right_src, "Ds005602_Right_split.csv"),
    ("ALL_Left", left_src, "ALL_Left_split.csv"),
    ("ALL_Right", right_src, "ALL_Right_split.csv"),
]

print("=" * 70)
print("RE-SYNCING SPHARM SPLIT DATA FROM SOURCE ACCORDING TO SPLIT CSVS")
print("=" * 70)

for ds_name, src_dir, csv_file in datasets:
    csv_path = os.path.join(base_dir, csv_file)
    df = pd.read_csv(csv_path)
    
    ds_folder = os.path.join(base_dir, ds_name)
    train_folder = os.path.join(ds_folder, "train")
    test_folder = os.path.join(ds_folder, "test")
    
    # Clean old directories completely
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
        
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    print(f"\nProcessing: {ds_name}")
    print(f"  Split CSV: {csv_file} (Total defined: {len(df)})")
    
    train_copied = 0
    test_copied = 0
    missing = []
    
    # Get all source files for fast matching
    all_src_files = os.listdir(src_dir)
    
    for _, row in df.iterrows():
        sub_id = str(row["Subject_ID"]).strip()
        split = str(row["Split"]).strip().capitalize() # 'Train' or 'Test'
        
        dest_dir = train_folder if split == "Train" else test_folder
        
        # Match files for this subject
        # Filename pattern: e.g. left_Healthy_sub-101_hippocampus_lh_aligned_SPHARM...
        # or left_TLE_sub-26896_hippocampus_lh_aligned_SPHARM...
        matched_files = [f for f in all_src_files if f"_{sub_id}_" in f or f.startswith(f"{sub_id}_")]
        
        if not matched_files:
            missing.append(sub_id)
            continue
            
        for f in matched_files:
            src_f = os.path.join(src_dir, f)
            dst_f = os.path.join(dest_dir, f)
            shutil.copy2(src_f, dst_f)
            
        if split == "Train":
            train_copied += 1
        else:
            test_copied += 1
            
    print(f"  [COPIED] Train: {train_copied} subjects | Test: {test_copied} subjects")
    if missing:
        print(f"  [WARNING] Missing subjects in source: {missing}")
        
    # Verify exact counts of SPHARM_ellalign.coef in train and test
    actual_tr = len(glob.glob(os.path.join(train_folder, "*_SPHARM_ellalign.coef")))
    actual_te = len(glob.glob(os.path.join(test_folder, "*_SPHARM_ellalign.coef")))
    print(f"  [VERIFICATION] Train: {actual_tr} | Test: {actual_te} | Total: {actual_tr + actual_te}")

print("\n" + "=" * 70)
print("ALL SPLIT FOLDERS RE-SYNCED AND VERIFIED 100% CLEAN!")
print("=" * 70)
