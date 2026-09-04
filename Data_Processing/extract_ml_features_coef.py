import os
import glob
import csv
import re
import argparse
import tkinter as tk
from tkinter import filedialog
import numpy as np


# =============================================================================
# Helper: Folder Picker
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "SPHARM", "split_data"))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Model", "Output_Dataset"))

def prompt_folder(title):
    init_dir = DEFAULT_DATA_DIR if os.path.exists(DEFAULT_DATA_DIR) else os.getcwd()
    try:
        import qt
        folder = qt.QFileDialog.getExistingDirectory(None, title, init_dir)
        return folder if folder else None
    except Exception:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title=title, initialdir=init_dir)
        root.destroy()
        return folder if folder else None


# =============================================================================
# Group Classification
# =============================================================================
def classify_subject(subject_name):
    is_left_side = subject_name.startswith("left_") or subject_name.startswith("lh_") or "_lh" in subject_name.lower()
    name_upper = subject_name.upper()
    
    # 1. Healthy Control / Normal (0)
    if "_HEALTHY" in name_upper or "HEALTHY" in name_upper or "HFH_" in name_upper or "NORMAL" in name_upper:
        return "Healthy Control", 0
    
    # 2. Ipsilateral TLE (Diseased) (1)
    elif (is_left_side and "LEFT-TLE" in name_upper) or (not is_left_side and "RIGHT-TLE" in name_upper):
        return "Ipsilateral TLE (Diseased)", 1
        
    # 3. Contralateral TLE (Healthy-side) (2)
    elif (is_left_side and "RIGHT-TLE" in name_upper) or (not is_left_side and "LEFT-TLE" in name_upper):
        return "Contralateral TLE (Healthy-side)", 2
        
    # 4. General TLE (1)
    elif "TLE" in name_upper:
        return "Ipsilateral TLE (Diseased)", 1
        
    return "Unknown", -1


# =============================================================================
# Parse SPHARM-PDM .coef format
# =============================================================================
def parse_coef(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Match all triplets {x, y, z}
    pattern = re.compile(r"\{([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+)\}")
    matches = pattern.findall(content)
    
    coeffs = []
    for m in matches:
        coeffs.append([float(x) for x in m])
    
    # Get num coeffs from first number e.g. { 169, ...
    num_match = re.search(r"\{\s*(\d+)", content)
    if num_match:
        num_coeffs = int(num_match.group(1))
        return coeffs[:num_coeffs]
    
    return coeffs


# =============================================================================
# Main Extraction
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Extract SPHARM coefficient features for Machine Learning.")
    parser.add_argument("--spharm_dir", "--input_dir", default=None,
                        help="Path to folder containing .coef files (or split dataset folder).")
    parser.add_argument("--output_dir", default=None,
                        help="Path to directory to save extracted CSVs (default: Model/Output_Dataset).")
    args = parser.parse_args()

    spharm_dir = args.spharm_dir
    if not spharm_dir:
        print("Opening folder picker...")
        spharm_dir = prompt_folder("Select SPHARM .coef folder (or dataset split folder)")
    
    if not spharm_dir:
        print("Error: No folder selected. Exiting.")
        return

    # Handle parent directory selection
    if os.path.basename(spharm_dir.rstrip("\\/")).lower() != "spharm_results":
        candidate = os.path.join(spharm_dir, "spharm_results")
        if os.path.isdir(candidate):
            spharm_dir = candidate

    print(f"Target folder: {spharm_dir}")

    # 1. ค้นหาไฟล์ .coef
    coef_files = sorted(glob.glob(os.path.join(spharm_dir, "*_SPHARM_ellalign.coef")))
    source = "ellalign"
    if not coef_files:
        coef_files = sorted(glob.glob(os.path.join(spharm_dir, "*_SPHARM.coef")))
        source = "spharm"
    if not coef_files:
        coef_files = sorted(glob.glob(os.path.join(spharm_dir, "*.coef")))
        source = "coef"

    if not coef_files:
        print(f"Error: No SPHARM .coef files found in {spharm_dir}")
        return

    print(f"Found {len(coef_files)} coefficient files (source type: '{source}'). Processing...")

    # 2. อ่านไฟล์แรกเพื่อทราบจำนวนสัมประสิทธิ์
    first_coeffs = parse_coef(coef_files[0])
    num_coeffs = len(first_coeffs)
    print(f"Number of SPHARM coefficients per subject: {num_coeffs} (total {num_coeffs * 3} values)")

    # 3. สร้างหัวตาราง (Header)
    header = ["Subject", "Group", "Class", "BinaryClass"] + [f"Coef_{i+1}" for i in range(num_coeffs * 3)]

    # ตั้งค่าโฟลเดอร์สำหรับผลลัพธ์ ML
    folder_basename = os.path.basename(spharm_dir.rstrip("\\/"))
    parent_dir = os.path.dirname(spharm_dir.rstrip("\\/"))
    parent_basename = os.path.basename(parent_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    grandparent_basename = os.path.basename(grandparent_dir)

    if folder_basename.lower() == "balanced" and parent_basename.lower() in ("train", "test"):
        name_prefix = f"{grandparent_basename}_{parent_basename}_balanced"
    elif folder_basename.lower() == "spharm_results":
        name_prefix = parent_basename
    elif folder_basename.lower() in ("train", "test", "balanced"):
        name_prefix = f"{parent_basename}_{folder_basename}"
    else:
        name_prefix = folder_basename

    if args.output_dir:
        ml_output_dir = os.path.abspath(args.output_dir)
    else:
        ml_output_dir = DEFAULT_OUTPUT_DIR

    os.makedirs(ml_output_dir, exist_ok=True)

    coef_csv_path = os.path.join(ml_output_dir, f"{name_prefix}_coef_features.csv")

    with open(coef_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for filepath in coef_files:
            filename = os.path.basename(filepath)
            
            # ลบส่วนขยายออกเพื่อใช้เป็นชื่อ Subject
            subject_name = filename
            for suffix in ("_SPHARM_ellalign.coef", "_SPHARM.coef"):
                subject_name = subject_name.replace(suffix, "")

            # จัดกลุ่ม
            group_name, group_label = classify_subject(subject_name)
            binary_class = 1 if group_label == 1 else 0

            # โหลดสัมประสิทธิ์
            coeffs = parse_coef(filepath)
            if len(coeffs) != num_coeffs:
                print(f"Warning: File {filename} has {len(coeffs)} coefficients instead of {num_coeffs}. Skipping.")
                continue

            flat_coeffs = np.array(coeffs).ravel() # ขนาด (3 * num_coeffs,)

            # ประกอบข้อมูลแถว
            row = [subject_name, group_name, group_label, binary_class]
            row.extend(["{:.8f}".format(val) for val in flat_coeffs])
            writer.writerow(row)

    print(f"Successfully saved SPHARM coefficient features for {len(coef_files)} subjects to: {coef_csv_path}")
    print("=" * 60)
    print("Done! You can use this CSV file for your ML models.")
    print("=" * 60)


if __name__ == "__main__":
    main()
