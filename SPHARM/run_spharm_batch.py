import os
import sys

def _bootstrap_slicer():
    try:
        import slicer
        return
    except ImportError:
        pass

    candidates = [
        os.environ.get("SLICER_EXE", ""),
        r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe",
        r"C:\Program Files\SlicerSALT 5.0.0\SlicerSALT.exe",
    ]
    slicer_exe = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            slicer_exe = cand
            break

    if not slicer_exe:
        print("[ERROR] SlicerSALT not found!")
        print("        Expected location: C:\\Program Files\\SlicerSALT 6.0.0\\SlicerSALT.exe")
        print("        Set env var SLICER_EXE or check your installation.")
        sys.exit(1)

    import subprocess
    script_path = os.path.abspath(__file__)
    cmd = [slicer_exe, "--no-main-window", "--no-splash",
           "--python-script", script_path] + sys.argv[1:]
    print(f"[INFO] No 'slicer' module in this Python. Re-launching via SlicerSALT:")
    print(f"       {slicer_exe}")
    sys.exit(subprocess.call(cmd))

_bootstrap_slicer()

import glob
import time
import argparse
import subprocess
from datetime import datetime
import slicer
import vtk
import numpy as np

def prompt_folder(title):
    try:
        import qt
        folder = qt.QFileDialog.getExistingDirectory(None, title)
        if not folder:
            return None
        return folder
    except Exception:
        return None

def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        for arg in sys.argv:
            if arg.endswith(".py") and os.path.isfile(arg):
                return os.path.dirname(os.path.abspath(arg))
        return os.getcwd()

SCRIPT_DIR = get_script_dir()

def sprint(msg, log_file=None):
    print(msg)
    sys.stdout.flush()
    if log_file:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {msg}\n")

def cleanup_subject_files(output_base_dir, basename):
    patterns = [
        f"{basename}_pp.*",
        f"{basename}_para.vtk", f"{basename}_paraMix.txt",
        f"{basename}_paraPhi.txt", f"{basename}_paraPhiHalf.txt",
        f"{basename}_paraTheta.txt",
        f"{basename}_surf.vtk",
        f"{basename}_SPHARM.vtk", f"{basename}_SPHARM.coef",
        f"{basename}_SPHARM_ellalign.vtk", f"{basename}_SPHARM_ellalign.coef",
        f"{basename}_SPHARM_procalign.vtk", f"{basename}_SPHARM_procalign.coef",
        f"{basename}_SPHARM_grid.vtk",
        f"{basename}_SPHARMMedialAxis.vtk",
        f"{basename}_MedialAxisScalars.csv",
    ]
    removed = 0
    for pat in patterns:
        for f in glob.glob(os.path.join(output_base_dir, pat)):
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass
    return removed

def run_cli_checked(module, params, step_name, log_file):
    cli_node = slicer.cli.run(module, None, params, wait_for_completion=True)
    status = cli_node.GetStatusString()
    if status not in ("Completed", "Completed with errors"):
        sprint(f"    !!! CLI '{step_name}' status = {status}", log_file)
        err = cli_node.GetErrorText() or ""
        out = cli_node.GetOutputText() or ""
        if err.strip():
            sprint(f"    stderr: {err.strip()[:1000]}", log_file)
        if out.strip():
            sprint(f"    stdout: {out.strip()[:1000]}", log_file)
        return False
    return True

def node_has_points(node, min_points=10):
    if node is None:
        return False
    pd = node.GetPolyData()
    if pd is None:
        return False
    return pd.GetNumberOfPoints() >= min_points

def improve_label_quality(input_node, log_file):
    try:
        from scipy import ndimage
        import numpy as _np
    except ImportError:
        sprint("    (skipping hole-fill: scipy not available in this Python)", log_file)
        return

    arr = slicer.util.arrayFromVolume(input_node)
    if arr is None or arr.size == 0:
        return
    binary = (arr > 0)
    before = int(binary.sum())
    if before == 0:
        return

    filled = ndimage.binary_fill_holes(binary)

    struct = ndimage.generate_binary_structure(3, 1)
    closed = ndimage.binary_closing(filled, structure=struct, iterations=1)

    labeled, n_comp = ndimage.label(closed, structure=struct)
    if n_comp > 1:
        sizes = ndimage.sum(closed, labeled, range(1, n_comp + 1))
        largest = int(_np.argmax(sizes)) + 1
        cleaned = (labeled == largest)
        removed_comps = n_comp - 1
    else:
        cleaned = closed
        removed_comps = 0

    after = int(cleaned.sum())
    new_arr = cleaned.astype(_np.uint8)
    slicer.util.updateVolumeFromArray(input_node, new_arr)

    delta = after - before
    if delta != 0 or removed_comps > 0:
        sprint(f"    Quality fix: {before} -> {after} voxels (delta={delta:+d})  "
               f"components removed: {removed_comps}", log_file)

def parse_args():
    parser = argparse.ArgumentParser(description="Batch SPHARM-PDM processing for SlicerSALT.")
    parser.add_argument("--input_dir", type=str, help="Directory containing .nii.gz labels")
    parser.add_argument("--output_dir", type=str, help="Directory to save all results")
    parser.add_argument("--fast", action="store_true", help="Fast/test mode: ~4-5x faster but lower mesh resolution")
    parser.add_argument("--regenerate_spharm_only", action="store_true", help="Skip SegPostProcess and GenParaMesh")
    parser.add_argument("--reference_template", type=str, default=None,
                        help="Path to reference _SPHARM.vtk used as regTemplate/flipTemplate.")
    parser.add_argument("--num_iterations", type=int, default=None, help="Iterations for GenParaMesh")
    parser.add_argument("--subdiv_level", type=int, default=None, help="Subdivision level for ParaToSPHARMMesh")
    parser.add_argument("--spharm_degree", type=int, default=None, help="SPHARM degree for ParaToSPHARMMesh")
    args, _ = parser.parse_known_args()
    return args

def init_logging(output_base_dir, input_dir, output_root, mode_tag, num_iter, subdiv, degree):
    log_file = os.path.join(SCRIPT_DIR, "spharm_debug_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"--- SPHARM Batch Mode Start: {datetime.now()} ---\n")
        f.write(f"Mode: {mode_tag} (iter={num_iter}, subdiv={subdiv}, degree={degree})\n")
        f.write(f"Input: {input_dir}\n")
        f.write(f"Output: {output_root}\n")
        f.write(f"SCRIPT_DIR: {SCRIPT_DIR}\n\n")
    sprint(f"SCRIPT_DIR resolved to: {SCRIPT_DIR}", log_file)
    return log_file

def find_label_files(input_dir, log_file):
    extensions = ["*.nii.gz", "*.nii", "*.hdr"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    file_list = sorted(list(set(file_list)))
    label_files = [f for f in file_list if "label" in os.path.basename(f).lower()]
    if label_files:
        sprint(f"Detected {len(label_files)} label-specific files (filtering out original images).", log_file)
        file_list = label_files

    sprint(f"Verified {len(file_list)} files for processing.", log_file)
    return file_list

def resolve_reference_template(args, file_list, output_base_dir, log_file):
    reference_template = args.reference_template
    template_basename = None

    if not reference_template and file_list:
        first_basename = os.path.basename(file_list[0]).split('.')[0]
        for suffix in ("_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
            candidate = os.path.join(output_base_dir, f"{first_basename}{suffix}").replace("\\", "/")
            if os.path.exists(candidate):
                reference_template = candidate
                template_basename = first_basename
                sprint(f"\nAuto-picked reference template: {os.path.basename(reference_template)}", log_file)
                break
        if not reference_template:
            sprint(f"\nNo existing SPHARM output for first subject — will process it without template first.", log_file)

    if reference_template:
        reference_template = reference_template.replace("\\", "/")
        if template_basename is None:
            tmpl_fname = os.path.basename(reference_template)
            for suf in ("_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
                if tmpl_fname.endswith(suf):
                    template_basename = tmpl_fname[: -len(suf)]
                    break
        sprint(f"Using regTemplate + flipTemplate = {reference_template}", log_file)
        sprint(f"Template subject basename: {template_basename}", log_file)

    return reference_template, template_basename

def process_single_subject(file_path, index, total_files, output_base_dir, args, config, template_info, log_file):
    file_path = os.path.normpath(file_path).replace("\\", "/")
    basename = os.path.basename(file_path).split('.')[0]
    sprint(f"\n>>> [{index+1}/{total_files}] STARTING: {basename}", log_file)

    pp_mask_path = os.path.join(output_base_dir, f"{basename}_pp.nrrd").replace("\\", "/")
    para_mesh_path = os.path.join(output_base_dir, f"{basename}_para.vtk").replace("\\", "/")
    surf_mesh_path = os.path.join(output_base_dir, f"{basename}_surf.vtk").replace("\\", "/")
    spharm_base = os.path.join(output_base_dir, basename).replace("\\", "/")

    reference_template, template_basename = template_info

    regen_mode = (args.regenerate_spharm_only
                  and os.path.exists(para_mesh_path)
                  and os.path.exists(surf_mesh_path))

    if not regen_mode:
        n_removed = cleanup_subject_files(output_base_dir, basename)
        if n_removed > 0:
            sprint(f"  - Cleaned {n_removed} stale files from previous run", log_file)
    else:
        for pat in [f"{basename}_SPHARM*.*", f"{basename}_SPHARMMedialAxis.vtk", f"{basename}_MedialAxisScalars.csv"]:
            for f in glob.glob(os.path.join(output_base_dir, pat)):
                try: os.remove(f)
                except OSError: pass

    input_node = None
    pp_node = None
    para_node = None
    surf_node = None

    new_template = None
    new_tmpl_basename = None

    try:
        if regen_mode:
            sprint(f"  - REGEN MODE: reusing _para.vtk + _surf.vtk", log_file)
            para_node = slicer.util.loadModel(para_mesh_path)
            surf_node = slicer.util.loadModel(surf_mesh_path)
            if not node_has_points(para_node) or not node_has_points(surf_node):
                sprint(f"  - !!! Failed to load _para/_surf — skipping", log_file)
                return template_info
        else:
            sprint(f"  - Loading volume...", log_file)
            input_node = slicer.util.loadLabelVolume(file_path)
            if not input_node:
                sprint(f"  - FAILED to load: {file_path}", log_file)
                return template_info

            sprint(f"  - STEP 0: Improving label quality (fill holes + close + largest comp)...", log_file)
            improve_label_quality(input_node, log_file)

            sprint(f"  - STEP 1: SegPostProcess (Cleaning labels)...", log_file)
            pp_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{basename}_pp")
            pp_params = {'fileName': input_node.GetID(), 'outfileName': pp_node.GetID(), 'label': 1}
            if not run_cli_checked(slicer.modules.segpostprocessclp, pp_params, "SegPostProcess", log_file):
                return template_info
            slicer.util.saveNode(pp_node, pp_mask_path)

            sprint(f"  - STEP 2: GenParaMesh (Creating spherical mesh)...", log_file)
            para_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{basename}_para")
            surf_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{basename}_surf")
            para_params = {
                'infile': pp_node.GetID(),
                'outParaName': para_node.GetID(),
                'outSurfName': surf_node.GetID(),
                'numIterations': config['num_iter'],
                'label': 1
            }
            if not run_cli_checked(slicer.modules.genparameshclp, para_params, "GenParaMesh", log_file):
                return template_info
            if not node_has_points(para_node) or not node_has_points(surf_node):
                sprint(f"  - !!! GenParaMesh returned EMPTY mesh — skipping", log_file)
                return template_info
            slicer.util.saveNode(para_node, para_mesh_path)
            slicer.util.saveNode(surf_node, surf_mesh_path)

        sprint(f"  - STEP 3: ParaToSPHARMMesh (Computing PDM)...", log_file)
        spharm_params = {
            'inParaFile': para_node.GetID(),
            'inSurfFile': surf_node.GetID(),
            'outbase': spharm_base,
            'subdivLevel': config['subdiv'],
            'spharmDegree': config['degree']
        }

        own_spharm_path = f"{spharm_base}_SPHARM.vtk"
        own_ellalign_path = f"{spharm_base}_SPHARM_ellalign.vtk"
        is_template_subject = (template_basename is not None and basename == template_basename)

        if reference_template and not is_template_subject:
            spharm_params['regTemplateFile'] = reference_template
            spharm_params['regTemplateFileOn'] = True
            spharm_params['flipTemplateFile'] = reference_template.replace(".vtk", ".coef")
            spharm_params['flipTemplateFileOn'] = True
            spharm_params['regTemplate'] = reference_template
            spharm_params['flipTemplate'] = reference_template.replace(".vtk", ".coef")
            spharm_params['flipTemplateOn'] = True
            sprint(f"    Using template alignment -> will produce _SPHARM_procalign.vtk", log_file)
        elif is_template_subject:
            sprint(f"    (Template subject — no self-reference)", log_file)

        run_cli_checked(slicer.modules.paratospharmmeshclp, spharm_params, "ParaToSPHARMMesh", log_file)

        if not (os.path.exists(own_spharm_path) and os.path.exists(own_spharm_path.replace(".vtk", ".coef"))):
            sprint(f"  - ERROR: SPHARM outputs not found for {basename}", log_file)
            cleanup_subject_files(output_base_dir, basename)
            return template_info

        try:
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(own_spharm_path)
            reader.Update()
            poly = reader.GetOutput()
            if poly is None or poly.GetNumberOfPoints() < 10 or poly.GetNumberOfCells() == 0:
                sprint(f"  - ERROR: Invalid/Corrupted SPHARM output (0 cells / NaN) for {basename}", log_file)
                cleanup_subject_files(output_base_dir, basename)
                return template_info
        except Exception:
            pass

        if is_template_subject and os.path.exists(own_ellalign_path):
            own_procalign_path = f"{spharm_base}_SPHARM_procalign.vtk"
            import shutil
            try:
                shutil.copy2(own_ellalign_path, own_procalign_path)
                sprint(f"    Copied template's ellalign -> procalign for pipeline uniformity", log_file)
            except Exception as e:
                sprint(f"    WARN: failed to copy ellalign->procalign: {e}", log_file)

        if not reference_template and os.path.exists(own_ellalign_path):
            new_template = own_ellalign_path
            new_tmpl_basename = basename
            sprint(f"    Set as reference template for remaining subjects: {os.path.basename(own_ellalign_path)}", log_file)

        final_vtk = f"{spharm_base}_SPHARM.vtk"
        final_coef = f"{spharm_base}_SPHARM.coef"
        grid_vtk = f"{spharm_base}_SPHARM_grid.vtk"

        if os.path.exists(final_vtk) and os.path.exists(final_coef):
            theta_step, phi_step = 4.5, 4.5
            sprint(f"  - SUCCESS: {basename} completed. Resampling to {theta_step}x{phi_step} degree grid...", log_file)

            try:
                if SCRIPT_DIR not in sys.path:
                    sys.path.insert(0, SCRIPT_DIR)
                import resample_spharm_grid
                coeffs = resample_spharm_grid.parse_coef(final_coef)
                L = int(np.sqrt(len(coeffs))) - 1
                theta_deg = np.linspace(0, 180, int(180/theta_step) + 1)
                phi_deg = np.linspace(0, 360, int(360/phi_step) + 1)[:-1]
                X, Y, Z = resample_spharm_grid.evaluate_spharm(coeffs, L, np.radians(theta_deg), np.radians(phi_deg))
                resample_spharm_grid.save_grid_vtk(X, Y, Z, theta_deg, phi_deg, grid_vtk)
                sprint(f"  - SUCCESS: {basename} Grid VTK created ({len(theta_deg)}x{len(phi_deg)} grid).", log_file)
            except Exception as resample_err:
                sprint(f"  - WARNING: Grid resampling failed for {basename}: {resample_err}", log_file)
        else:
            sprint(f"  - ERROR: Result VTK not generated for {basename}.", log_file)

    except Exception as e:
        import traceback
        sprint(f"  - CRITICAL ERROR for {basename}: {str(e)}", log_file)
        sprint(traceback.format_exc(), log_file)

    finally:
        for node in [input_node, pp_node, para_node, surf_node]:
            if node:
                try: slicer.mrmlScene.RemoveNode(node)
                except Exception: pass
        sprint(f"  - Cleanup done.", log_file)

    if new_template:
        return (new_template, new_tmpl_basename)
    return template_info

def run_batch_spharm():
    args = parse_args()

    if args.num_iterations is not None or args.subdiv_level is not None or args.spharm_degree is not None:
        NUM_ITER = args.num_iterations if args.num_iterations is not None else (200 if args.fast else 1000)
        SUBDIV   = args.subdiv_level if args.subdiv_level is not None else (5 if args.fast else 10)
        DEGREE   = args.spharm_degree if args.spharm_degree is not None else (6 if args.fast else 12)
        MODE_TAG = "CUSTOM"
    elif args.fast:
        NUM_ITER, SUBDIV, DEGREE, MODE_TAG = 200, 5, 6, "FAST"
    else:
        NUM_ITER, SUBDIV, DEGREE, MODE_TAG = 1000, 10, 12, "PRODUCTION"

    input_dir = os.path.abspath(args.input_dir) if args.input_dir else None
    output_root = os.path.abspath(args.output_dir) if args.output_dir else None

    if not input_dir:
        cand_default = os.path.join(SCRIPT_DIR, "output", "aligned_nifti")
        cand_icp = os.path.join(os.path.dirname(SCRIPT_DIR), "ICP", "output", "aligned_nifti")

        if os.path.isdir(cand_default) and len(glob.glob(os.path.join(cand_default, "*.nii*"))) > 0:
            input_dir = cand_default
            print(f"[INFO] Auto-detected input aligned NIfTI directory: {input_dir}")
        elif os.path.isdir(cand_icp) and len(glob.glob(os.path.join(cand_icp, "*.nii*"))) > 0:
            input_dir = cand_icp
            print(f"[INFO] Auto-detected ICP aligned NIfTI directory: {input_dir}")
        else:
            print("[INFO] No --input_dir specified. Opening folder picker dialog...")
            input_dir = prompt_folder("Select input folder containing aligned NIfTI labels (.nii.gz)")
            if not input_dir:
                print("[ERROR] No input folder selected. Exiting.")
                return

    if not output_root:
        if os.path.basename(os.path.normpath(input_dir)) == "aligned_nifti":
            output_root = os.path.dirname(os.path.normpath(input_dir))
        else:
            output_root = os.path.join(SCRIPT_DIR, "output")

    output_base_dir = os.path.join(output_root, "spharm_results")
    os.makedirs(output_base_dir, exist_ok=True)

    log_file = init_logging(output_base_dir, input_dir, output_root, MODE_TAG, NUM_ITER, SUBDIV, DEGREE)
    file_list = find_label_files(input_dir, log_file)
    if not file_list:
        sprint(f"[ERROR] No valid NIfTI label files found in {input_dir}", log_file)
        return

    template_info = resolve_reference_template(args, file_list, output_base_dir, log_file)

    config = {'num_iter': NUM_ITER, 'subdiv': SUBDIV, 'degree': DEGREE}

    for i, file_path in enumerate(file_list):
        template_info = process_single_subject(file_path, i, len(file_list), output_base_dir, args, config, template_info, log_file)

    sprint("\n" + "="*60, log_file)
    sprint("!!! ALL BATCH PROCESSING COMPLETED !!!", log_file)
    sprint("="*60, log_file)

if __name__ == "__main__":
    exit_code = 0
    try:
        run_batch_spharm()
    except Exception as _e:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            slicer.util.exit(exit_code)
        except Exception:
            pass
        try:
            import time, threading
            def force_kill():
                time.sleep(1.5)
                os._exit(exit_code)
            t = threading.Thread(target=force_kill, daemon=True)
            t.start()
        except Exception:
            os._exit(exit_code)