import vtk
from vtk.util import numpy_support
import os
import sys
import glob
import csv
import re
import shutil
import argparse
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from scipy.spatial import ConvexHull
import scipy.special as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPHARM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "SPHARM"))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(SPHARM_DIR, "split_data"))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SPHARM_DIR not in sys.path:
    sys.path.insert(0, SPHARM_DIR)

try:
    from resample_spharm_grid import evaluate_spharm, save_grid_vtk
except ImportError:
    pass


# =============================================================================
# Icosahedron Subdivision Logic (1002 points)
# =============================================================================
def get_spharm_template_parameters(input_dir=None):
    # Find any standard SPHARM-PDM ellalign vtk to extract standard 1002-point topology, thetas, and phis
    template_candidates = []
    
    if input_dir and os.path.isdir(input_dir):
        template_candidates.extend(glob.glob(os.path.join(input_dir, "**", "*_SPHARM_ellalign.vtk"), recursive=True))
        template_candidates.extend(glob.glob(os.path.join(input_dir, "**", "*_SPHARM.vtk"), recursive=True))

    workspace_root = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    
    search_patterns = [
        os.path.join(SPHARM_DIR, "split_data", "**", "*_SPHARM_ellalign.vtk"),
        os.path.join(SPHARM_DIR, "split_data", "**", "*_SPHARM.vtk"),
        os.path.join(workspace_root, "ICP", "output_*", "spharm_results_*", "*_SPHARM_ellalign.vtk"),
        os.path.join(workspace_root, "ICP", "output_*", "*_SPHARM_ellalign.vtk"),
        os.path.join(workspace_root, "**", "*_SPHARM_ellalign.vtk"),
        os.path.join(workspace_root, "**", "*_SPHARM.vtk"),
    ]
    
    for pat in search_patterns:
        if template_candidates:
            break
        template_candidates.extend(glob.glob(pat, recursive=True))

    for candidate in template_candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 1000:
            try:
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(candidate)
                reader.Update()
                poly = reader.GetOutput()
                if poly and poly.GetNumberOfPoints() > 0 and poly.GetPointData().GetArray('_paraTheta'):
                    thetas = numpy_support.vtk_to_numpy(poly.GetPointData().GetArray('_paraTheta'))
                    phis = numpy_support.vtk_to_numpy(poly.GetPointData().GetArray('_paraPhi'))
                    
                    cells = poly.GetPolys()
                    id_list = vtk.vtkIdList()
                    cells.InitTraversal()
                    triangles = []
                    while cells.GetNextCell(id_list):
                        if id_list.GetNumberOfIds() == 3:
                            triangles.append((id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)))
                    triangles = np.array(triangles)
                    return None, triangles, thetas, phis
            except Exception:
                continue

    # Fallback to mathematical icosahedron if no template vtk found
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
    ])
    verts = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    all_points = []
    N = 10
    for face in faces:
        A, B, C = verts[face[0]], verts[face[1]], verts[face[2]]
        for i in range(N + 1):
            for j in range(N + 1 - i):
                k = N - i - j
                P = (i/N)*A + (j/N)*B + (k/N)*C
                all_points.append(P / np.linalg.norm(P))
    unique_points = np.unique(np.round(np.array(all_points), 6), axis=0)
    hull = ConvexHull(unique_points)
    X, Y, Z = unique_points[:, 0], unique_points[:, 1], unique_points[:, 2]
    thetas = np.arccos(np.clip(Z, -1.0, 1.0))
    phis = np.arctan2(Y, X)
    phis = np.where(phis < 0, phis + 2*np.pi, phis)
    return unique_points, hull.simplices, thetas, phis

def evaluate_spharm_icosahedron(coeffs_list, L, thetas, phis):
    X = np.zeros_like(thetas)
    Y = np.zeros_like(thetas)
    Z = np.zeros_like(thetas)
    
    idx = 0
    for l in range(L + 1):
        c = coeffs_list[idx]
        y = sp.sph_harm_y(l, 0, thetas, phis).real
        X += c[0] * y
        Y += c[1] * y
        Z += c[2] * y
        idx += 1
        
        for m in range(1, l + 1):
            cr = coeffs_list[idx]
            idx += 1
            ci = coeffs_list[idx]
            idx += 1
            
            y_comp = sp.sph_harm_y(l, m, thetas, phis)
            factor = np.sqrt(2)
            
            X += factor * (cr[0] * y_comp.real + ci[0] * y_comp.imag)
            Y += factor * (cr[1] * y_comp.real + ci[1] * y_comp.imag)
            Z += factor * (cr[2] * y_comp.real + ci[2] * y_comp.imag)
            
    return X, Y, Z

def save_polydata_vtk(X, Y, Z, triangles, thetas, phis, output_path):
    num_points = len(X)
    
    points = vtk.vtkPoints()
    theta_arr = vtk.vtkFloatArray()
    theta_arr.SetName("_paraTheta")
    phi_arr = vtk.vtkFloatArray()
    phi_arr.SetName("_paraPhi")
    
    for i in range(num_points):
        points.InsertNextPoint(X[i], Y[i], Z[i])
        theta_arr.InsertNextValue(thetas[i])
        phi_arr.InsertNextValue(phis[i])
        
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.GetPointData().AddArray(theta_arr)
    poly.GetPointData().AddArray(phi_arr)
    
    cells = vtk.vtkCellArray()
    for tri in triangles:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, tri[0])
        triangle.GetPointIds().SetId(1, tri[1])
        triangle.GetPointIds().SetId(2, tri[2])
        cells.InsertNextCell(triangle)
        
    poly.SetPolys(cells)
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(poly)
    writer.Write()

# =============================================================================
# Helper: Folder Picker
# =============================================================================
def prompt_folder(title):
    init_dir = DEFAULT_DATA_DIR if os.path.exists(DEFAULT_DATA_DIR) else os.getcwd()
    try:
        import qt
        folder = qt.QFileDialog.getExistingDirectory(None, title, init_dir)
        return folder if folder else None
    except Exception:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title=title, initialdir=init_dir)
        root.destroy()
        return folder if folder else None

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
    
    num_match = re.search(r"\{\s*(\d+)", content)
    if num_match:
        num_coeffs = int(num_match.group(1))
        return coeffs[:num_coeffs]
    
    return coeffs

# =============================================================================
# Write SPHARM-PDM .coef format
# =============================================================================
def save_coef(coeffs, filepath):
    num_coeffs = len(coeffs)
    with open(filepath, 'w') as f:
        # Write first line
        f.write(f"{{ {num_coeffs},{{{coeffs[0][0]:.6f}, {coeffs[0][1]:.6f}, {coeffs[0][2]:.6f}}},\n")
        # Write middle lines
        for i in range(1, num_coeffs - 1):
            f.write(f"{{{coeffs[i][0]:.6f}, {coeffs[i][1]:.6f}, {coeffs[i][2]:.6f}}},\n")
        # Write last line
        f.write(f"{{{coeffs[-1][0]:.6f}, {coeffs[-1][1]:.6f}, {coeffs[-1][2]:.6f}}}}}")

# =============================================================================
# Group Classification (binary: Healthy=0, TLE=1)
# =============================================================================
def classify_subject(subject_name):
    name_upper = subject_name.upper()
    
    # Healthy Control / Normal (0)
    if "_HEALTHY" in name_upper or "HEALTHY" in name_upper or "HFH_" in name_upper or "NORMAL" in name_upper:
        return "Healthy", 0
    
    # TLE (1) — includes Left-TLE, Right-TLE, general TLE
    if "TLE" in name_upper:
        return "TLE", 1
        
    return "Unknown", -1


def main():
    print("=" * 60)
    print("--- Balanced PLS-DA Interpolation Augmentation ---")
    print("  Augments minority class to match majority class count")
    print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--input_dir", "--spharm_dir", type=str, default=None,
                        help="Directory containing *_SPHARM.coef files (or split dataset folder e.g. Ds004469_Left/train)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Optional specific directory to save the output (e.g. on Desktop). If not given, creates a 'balanced' subfolder.")
    parser.add_argument("--n_components", type=int, default=10,
                        help="Number of PLS-DA components to fit (default=10)")
    parser.add_argument("--num_per_pair", type=int, default=None,
                        help="Number of children (interpolations) to generate per parent pair (prompts if None)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args, unknown = parser.parse_known_args()

    np.random.seed(args.seed)

    # Interactive prompt if num_per_pair is not provided via CLI
    num_per_pair = args.num_per_pair
    if num_per_pair is None:
        while True:
            try:
                user_input = input("Enter the number of children to generate per parent pair (e.g. 8): ")
                num_per_pair = int(user_input)
                if num_per_pair <= 0:
                    print("Please enter a positive integer.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

    # -------------------------------------------------------------------------
    # 1. Select input directory
    # -------------------------------------------------------------------------
    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        print("No --output_dir given. Opening folder picker...")
        chosen = prompt_folder("Select dataset folder (containing *_SPHARM.coef files)")
        if not chosen:
            print("ERROR: No folder selected. Exiting.")
            return
        if os.path.basename(chosen.rstrip("\\/")) == "spharm_results":
            output_root = os.path.abspath(os.path.dirname(chosen))
        else:
            output_root = os.path.abspath(chosen)

    # Check if selected folder is directly the directory containing coef files, or contains 'spharm_results'
    if os.path.isdir(os.path.join(output_root, "spharm_results")):
        coef_dir = os.path.join(output_root, "spharm_results")
    else:
        coef_dir = output_root

    # -------------------------------------------------------------------------
    # 2. Find and parse .coef files
    # -------------------------------------------------------------------------
    all_ell_files = sorted(glob.glob(os.path.join(coef_dir, "*_SPHARM_ellalign.coef")))
    if all_ell_files:
        coef_files = all_ell_files
    else:
        all_coef_files = sorted(glob.glob(os.path.join(coef_dir, "*_SPHARM.coef")))
        coef_files = [f for f in all_coef_files
                      if not any(s in os.path.basename(f)
                                 for s in ("_ellalign", "_grid", "_realigned", "_procalign", "_pca_ready", "_interp_"))]

    if not coef_files:
        print(f"ERROR: No SPHARM coefficient files (*_SPHARM.coef) found in: {coef_dir}")
        return

    # First pass: determine expected coefficient length
    L = None
    expected_len = None
    for fpath in coef_files:
        coeffs = parse_coef(fpath)
        if coeffs and len(coeffs) >= 9:
            expected_len = len(coeffs)
            L = int(np.sqrt(expected_len)) - 1
            break

    if expected_len is None:
        print("ERROR: No valid SPHARM coefficient files found.")
        return

    # Second pass: read all files
    subject_names = []
    coef_vectors = []
    classes = []
    group_names = []
    file_paths = []

    for fpath in coef_files:
        basename = os.path.basename(fpath)
        coeffs = parse_coef(fpath)
        if not coeffs or len(coeffs) != expected_len:
            continue
        
        flat_coeffs = np.array(coeffs).ravel()
        coef_vectors.append(flat_coeffs)
        
        subj_name = basename.replace("_SPHARM_ellalign.coef", "").replace("_SPHARM.coef", "")
        subject_names.append(subj_name)
        g_name, cls = classify_subject(basename)
        group_names.append(g_name)
        classes.append(cls)
        file_paths.append(fpath)

    coef_vectors = np.array(coef_vectors)
    classes = np.array(classes)
    N, D = coef_vectors.shape
    print(f"\nLoaded {N} subjects. Coef vector size: {D}")

    # -------------------------------------------------------------------------
    # 3. Determine class distribution
    # -------------------------------------------------------------------------
    unique_classes, class_counts = np.unique(classes, return_counts=True)
    
    # Filter out unknown class (-1) if present
    valid_mask = unique_classes >= 0
    unique_classes = unique_classes[valid_mask]
    class_counts = class_counts[valid_mask]

    if len(unique_classes) < 2:
        print("ERROR: Need at least 2 classes to balance. Found only:", unique_classes)
        return

    print("\n--- Class Distribution (Before Augmentation) ---")
    class_label_map = {0: "Healthy", 1: "TLE"}
    for cls, cnt in zip(unique_classes, class_counts):
        print(f"  Class {cls} ({class_label_map.get(cls, 'Unknown')}): {cnt} subjects")

    majority_cls = unique_classes[np.argmax(class_counts)]
    minority_cls = unique_classes[np.argmin(class_counts)]
    n_majority = int(class_counts.max())
    n_minority = int(class_counts.min())

    print(f"\nMajority class: {class_label_map.get(majority_cls, majority_cls)} ({n_majority})")
    print(f"Minority class: {class_label_map.get(minority_cls, minority_cls)} ({n_minority})")
    print(f"Children per pair: {num_per_pair}")

    # -------------------------------------------------------------------------
    # 4. Fit PLS-DA on ALL data
    # -------------------------------------------------------------------------
    Y = np.zeros((N, 2))
    for i, cls in enumerate(classes):
        col = 1 if cls == 1 else 0
        Y[i, col] = 1.0

    n_comp = min(args.n_components, N - 1)
    print(f"\nFitting PLS-DA model with {n_comp} components...")
    pls = PLSRegression(n_components=n_comp, scale=True)
    X_scores, _ = pls.fit_transform(coef_vectors, Y)

    # -------------------------------------------------------------------------
    # 5. Form pairs WITHIN each class separately
    # -------------------------------------------------------------------------
    def form_pairs_within_class(indices):
        """Form pairs using nearest-neighbor in PLS-DA score space."""
        formed = []
        temp_pool = list(indices)
        np.random.shuffle(temp_pool)
        
        while len(temp_pool) >= 2:
            idx_A = temp_pool.pop(0)
            distances = [np.linalg.norm(X_scores[idx_A] - X_scores[cand]) for cand in temp_pool]
            min_pos = np.argmin(distances)
            idx_B = temp_pool.pop(min_pos)
            formed.append((idx_A, idx_B))
        return formed

    majority_indices = np.where(classes == majority_cls)[0].tolist()
    minority_indices = np.where(classes == minority_cls)[0].tolist()

    majority_pairs = form_pairs_within_class(majority_indices)
    minority_pairs = form_pairs_within_class(minority_indices)

    # Augment both classes with num_per_pair children per pair
    n_aug_majority = len(majority_pairs) * num_per_pair
    n_aug_minority = len(minority_pairs) * num_per_pair

    total_majority = n_majority + n_aug_majority
    total_minority = n_minority + n_aug_minority

    # Calculate extra augmentation needed for minority class to balance
    n_extra_minority = 0
    if total_majority > total_minority:
        n_extra_minority = total_majority - total_minority
    elif total_minority > total_majority:
        # Minority class ended up larger — trim by reducing augmentation
        # We simply cap minority augmentation so totals match
        n_aug_minority = total_majority - n_minority
        n_extra_minority = 0
        total_minority = total_majority

    total_target = max(total_majority, total_minority)
    n_total_aug_minority = n_aug_minority + n_extra_minority

    print(f"\n--- Augmentation Plan ---")
    print(f"  Majority ({class_label_map.get(majority_cls, majority_cls)}): {n_majority} original + {n_aug_majority} augmented = {total_majority}")
    print(f"  Minority ({class_label_map.get(minority_cls, minority_cls)}): {n_minority} original + {n_aug_minority} augmented" + 
          (f" + {n_extra_minority} extra" if n_extra_minority > 0 else "") + f" = {total_target}")
    print(f"  Final balanced total: {total_target} per class ({total_target * 2} total)")

    # Distribute children across minority pairs including extra
    # First: num_per_pair per pair for minority, then distribute extra across pairs
    minority_children_per_pair = [num_per_pair] * len(minority_pairs)
    if n_extra_minority > 0 and len(minority_pairs) > 0:
        extra_base = n_extra_minority // len(minority_pairs)
        extra_rem = n_extra_minority % len(minority_pairs)
        for i in range(len(minority_pairs)):
            minority_children_per_pair[i] += extra_base + (1 if i < extra_rem else 0)

    # For majority: cap total augmented to n_aug_majority
    majority_children_per_pair = [num_per_pair] * len(majority_pairs)

    print(f"\n  Majority pairs: {len(majority_pairs)} x {num_per_pair} children = {n_aug_majority}")
    print(f"  Minority pairs: {len(minority_pairs)} x {num_per_pair} children = {n_aug_minority}" +
          (f" + {n_extra_minority} extra" if n_extra_minority > 0 else ""))

    # -------------------------------------------------------------------------
    # 6. Setup output directory
    # -------------------------------------------------------------------------
    if args.save_dir:
        # Create a subfolder inside save_dir named after the input folder (e.g., All_left_Train)
        folder_name = os.path.basename(coef_dir.rstrip("\\/"))
        if folder_name.lower() == "spharm_results":
            folder_name = os.path.basename(os.path.dirname(coef_dir.rstrip("\\/")))
        
        balanced_dir = os.path.join(args.save_dir, folder_name)
    else:
        # Create 'balanced' directory inside the input directory
        balanced_dir = os.path.join(coef_dir, "balanced")
        
    # Clear previous contents to avoid stale files from prior runs
    if os.path.isdir(balanced_dir):
        shutil.rmtree(balanced_dir)
    os.makedirs(balanced_dir)

    # -------------------------------------------------------------------------
    # 7. Load 1002-point template parameters and copy original files
    # -------------------------------------------------------------------------
    _, triangles, thetas, phis = get_spharm_template_parameters(coef_dir)

    print(f"\nCopying {N} original .coef and .vtk files to: {balanced_dir}")
    for fpath in file_paths:
        dst = os.path.join(balanced_dir, os.path.basename(fpath))
        shutil.copy2(fpath, dst)
        vtk_src = fpath.replace(".coef", ".vtk")
        vtk_dst = os.path.join(balanced_dir, os.path.basename(vtk_src))
        if os.path.exists(vtk_src):
            shutil.copy2(vtk_src, vtk_dst)
        else:
            coeffs_orig = parse_coef(fpath)
            X_o, Y_o, Z_o = evaluate_spharm_icosahedron(coeffs_orig, L, thetas, phis)
            save_polydata_vtk(X_o, Y_o, Z_o, triangles, thetas, phis, vtk_dst)

    # -------------------------------------------------------------------------
    # 8. Generate synthetic samples via interpolation
    # -------------------------------------------------------------------------
    metadata_rows = []
    global_idx = 1

    def generate_children(pairs_list, children_list, class_cls, label):
        """Generate interpolated children for a list of pairs."""
        nonlocal global_idx
        total_for_class = sum(children_list)
        count = 0
        for pair_idx, (idx_A, idx_B) in enumerate(pairs_list):
            n_children = children_list[pair_idx]
            if n_children == 0:
                continue

            # Generate evenly spaced alphas with small random jitter
            if n_children == 1:
                alphas = np.array([0.5])
            else:
                alphas = np.linspace(0.1, 0.9, n_children)
                jitter = np.random.uniform(-0.02, 0.02, n_children)
                alphas = np.clip(alphas + jitter, 0.05, 0.95)

            for child_idx, alpha in enumerate(alphas):
                # Interpolate in PLS-DA score space
                score_A = X_scores[idx_A]
                score_B = X_scores[idx_B]
                score_new = (1 - alpha) * score_A + alpha * score_B

                # Reconstruct coefficients via inverse transform
                flat_recon = pls.inverse_transform(score_new.reshape(1, -1))[0]
                coeffs_recon = flat_recon.reshape(expected_len, 3)

                # Determine naming from parent A
                parent_name = subject_names[idx_A]
                
                # Determine side prefix
                if parent_name.startswith("lh_") or "_lh_" in parent_name.lower():
                    side_str = "lh_"
                elif parent_name.startswith("rh_") or "_rh_" in parent_name.lower():
                    side_str = "rh_"
                elif parent_name.lower().startswith("left_"):
                    side_str = "left_"
                elif parent_name.lower().startswith("right_"):
                    side_str = "right_"
                else:
                    side_str = ""

                # Determine class prefix from parent A's naming
                class_prefix = group_names[idx_A]

                # Save .coef file
                coef_filename = f"{side_str}{class_prefix}_aug_{global_idx:04d}_SPHARM_ellalign.coef"
                coef_filepath = os.path.join(balanced_dir, coef_filename)
                save_coef(coeffs_recon, coef_filepath)

                # Evaluate SPHARM on 1002-point icosahedron and save .vtk file
                X_grid, Y_grid, Z_grid = evaluate_spharm_icosahedron(
                    coeffs_recon, L, thetas, phis
                )
                vtk_filename = f"{side_str}{class_prefix}_aug_{global_idx:04d}_SPHARM_ellalign.vtk"
                vtk_filepath = os.path.join(balanced_dir, vtk_filename)
                save_polydata_vtk(X_grid, Y_grid, Z_grid, triangles, thetas, phis, vtk_filepath)
 
                # Log metadata
                metadata_rows.append([
                    coef_filename,
                    class_label_map.get(class_cls, str(class_cls)),
                    subject_names[idx_A],
                    group_names[idx_A],
                    subject_names[idx_B],
                    group_names[idx_B],
                    f"{alpha:.4f}",
                ])

                count += 1
                if count % 20 == 0 or count == total_for_class:
                    print(f"    Progress: {count}/{total_for_class}")
                global_idx += 1

    n_total_generate = n_aug_majority + n_total_aug_minority
    print(f"\nGenerating {n_total_generate} total synthetic samples...")

    # Generate for majority class
    print(f"\n  [{class_label_map.get(majority_cls, majority_cls)}] Generating {n_aug_majority} samples...")
    generate_children(majority_pairs, majority_children_per_pair, majority_cls, "majority")

    # Generate for minority class (including extra to balance)
    print(f"\n  [{class_label_map.get(minority_cls, minority_cls)}] Generating {n_total_aug_minority} samples...")
    generate_children(minority_pairs, minority_children_per_pair, minority_cls, "minority")

    # -------------------------------------------------------------------------
    # 9. Write metadata CSV
    # -------------------------------------------------------------------------
    metadata_csv_path = os.path.join(balanced_dir, "augmentation_metadata.csv")
    with open(metadata_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename", "Class", "Parent_A_Subject", "Parent_A_Group",
            "Parent_B_Subject", "Parent_B_Group", "Alpha"
        ])
        writer.writerows(metadata_rows)

    # -------------------------------------------------------------------------
    # 10. Write balance summary
    # -------------------------------------------------------------------------
    summary_path = os.path.join(balanced_dir, "balance_summary.txt")
    
    # Count final distribution
    final_coef_files = glob.glob(os.path.join(balanced_dir, "*.coef"))
    final_healthy = 0
    final_tle = 0
    for f in final_coef_files:
        _, cls = classify_subject(os.path.basename(f))
        if cls == 0:
            final_healthy += 1
        elif cls == 1:
            final_tle += 1

    with open(summary_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Balanced Augmentation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("BEFORE Augmentation:\n")
        for cls, cnt in zip(unique_classes, class_counts):
            f.write(f"  {class_label_map.get(cls, cls)}: {cnt}\n")
        f.write(f"  Total: {N}\n\n")
        f.write(f"Children per pair: {num_per_pair}\n")
        f.write(f"Synthetic samples generated: {n_total_generate}\n\n")
        f.write("AFTER Augmentation:\n")
        f.write(f"  Healthy: {final_healthy}\n")
        f.write(f"  TLE:     {final_tle}\n")
        f.write(f"  Total:   {final_healthy + final_tle}\n\n")
        f.write(f"Output directory: {os.path.abspath(balanced_dir)}\n")

    # -------------------------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Balanced Augmentation Complete!")
    print("=" * 60)
    print(f"\n  BEFORE:  Healthy={int(class_counts[unique_classes == 0][0])}  |  TLE={int(class_counts[unique_classes == 1][0])}  |  Total={N}")
    print(f"  AFTER:   Healthy={final_healthy}  |  TLE={final_tle}  |  Total={final_healthy + final_tle}")
    print(f"\n  Children per pair: {num_per_pair}")
    print(f"  Synthetic samples generated: {n_total_generate}")
    print(f"  Metadata CSV:  {metadata_csv_path}")
    print(f"  Summary file:  {summary_path}")
    print(f"  Output dir:    {os.path.abspath(balanced_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
