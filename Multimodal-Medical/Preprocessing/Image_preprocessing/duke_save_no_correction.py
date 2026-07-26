import os
import sys
import importlib
import importlib.metadata as importlib_metadata
import pandas as pd
import numpy as np
from PIL import Image


def _import_pydicom():
    """Import pydicom while skipping the Windows-blocked libjpeg plugin."""
    original_entry_points = importlib_metadata.entry_points

    def filtered_entry_points(*args, **kwargs):
        eps = original_entry_points(*args, **kwargs)
        if kwargs.get("group") != "pylibjpeg.pixel_data_decoders":
            return eps

        return [ep for ep in eps if getattr(ep, "module", None) != "libjpeg"]

    importlib_metadata.entry_points = filtered_entry_points
    try:
        # A prior failed import can leave pydicom submodules behind and
        # cause a circular-import style failure on retry in the same process.
        stale_modules = [
            name for name in sys.modules if name == "pydicom" or name.startswith("pydicom.")
        ]
        for name in stale_modules:
            sys.modules.pop(name, None)

        return importlib.import_module("pydicom")
    finally:
        importlib_metadata.entry_points = original_entry_points


pydicom = _import_pydicom()

def extract_3_slice_stack(patient_list, root_dir, anno_path, output_dir):
    dicom_module = globals().get("pydicom")
    if dicom_module is None:
        dicom_module = _import_pydicom()

    annotations = pd.read_excel(anno_path)
    
    for patient_id in patient_list:
        print(f"--- Processing Patient: {patient_id} ---")
        anno = annotations[annotations['PatientID'] == patient_id]
        if anno.empty: 
            continue
        
        # Extract coordinates from annotations
        r1, r2, c1, c2, s1, s2 = (anno['StartRow'].iloc[0], anno['EndRow'].iloc[0], 
                                  anno['StartColumn'].iloc[0], anno['EndColumn'].iloc[0], 
                                  anno['StartSlice'].iloc[0], anno['EndSlice'].iloc[0])

        # 1. SEARCH: Scan the ROOT for DICOM files belonging to this Patient ID
        target_files = []
        for root, dirs, files in os.walk(root_dir):
            if str(patient_id) in root:
                for f in files:
                    full_p = os.path.join(root, f)
                    # Filter for actual image files by size
                    if os.path.getsize(full_p) > 50000:
                        target_files.append(full_p)

        if not target_files:
            print(f"  [FATAL] {patient_id}: No valid image files found.")
            continue

        # 2. Extract raw pixel arrays
        valid_arrays = []
        first_decode_error = None
        for f_path in target_files:
            try:
                ds = dicom_module.dcmread(f_path, force=True)
                valid_arrays.append((ds.pixel_array, f_path))
                if len(valid_arrays) > 500: break # Optimization
            except Exception as exc:
                if first_decode_error is None:
                    first_decode_error = exc
                continue

        if not valid_arrays:
            print(f"  [FATAL] {patient_id}: Files found, but none could be decoded.")
            if first_decode_error is not None:
                print(f"    First decode error: {first_decode_error}")
            continue

        # Sort by filename to maintain anatomical sequence
        valid_arrays.sort(key=lambda x: x[1])

        # 3. Select Target Slice
        # Use middle of annotated range, clamped to available slices
        target_idx = int(s1 + (s2 - s1) // 2) - 1
        idx = max(0, min(target_idx, len(valid_arrays)-1))
        
        # Get raw data (No N4 or Otsu processing)
        raw_arr = valid_arrays[idx][0].astype(np.float32)
        
        try:
            # 4. ROI Cropping (256x256)
            center_r, center_c = int((r1 + r2) // 2), int((c1 + c2) // 2)
            half_size = 128
            
            # Pad to handle cases where the tumor is near the edge of the image
            padded = np.pad(raw_arr, pad_width=half_size, mode='constant', constant_values=0)
            adj_r, adj_c = center_r + half_size, center_c + half_size
            crop = padded[adj_r - half_size : adj_r + half_size, adj_c - half_size : adj_c + half_size]

            # 5. Intensity Normalization (The "BreastDCEDL" Strategy)
            # Clip the top/bottom 1% to handle outliers and scale to 0-255
            vmin, vmax = np.percentile(crop, [1, 99])
            if vmax - vmin == 0: vmax += 1e-5 # Prevent division by zero
            
            crop_norm = np.clip(crop, vmin, vmax)
            crop_norm = ((crop_norm - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            
            final_img = Image.fromarray(crop_norm)

            # 6. Save results
            save_path = os.path.join(output_dir, str(patient_id))
            os.makedirs(save_path, exist_ok=True)
            for i in range(1, 4):
                # Saved as PNG to preserve normalized 8-bit values
                final_img.save(os.path.join(save_path, f"slice_{i}_raw_normalized_256.png"))

            print(f"  [SUCCESS] {patient_id} processed (Raw + Percentile Normalization).")
            
        except Exception as e:
            print(f"  [ERROR] {patient_id}: {e}")

# --- Paths ---
PT_LIST = "C:/Users/y0qz1/Desktop/dataset_patients_list_all.xlsx"
ROOT = "D:/Duke-Breast-Cancer-MRI" 
ANNO_FILE = "C:/Users/y0qz1/Desktop/Annotation_Boxes.xlsx"
OUT = "C:/Users/y0qz1/Desktop/Extracted_Slices_Raw_NoN4"

# --- Execution ---
if __name__ == "__main__":
    pts = pd.read_excel(PT_LIST)
    patients = list(pts["PatientID"])
    extract_3_slice_stack(patients, ROOT, ANNO_FILE, OUT)
