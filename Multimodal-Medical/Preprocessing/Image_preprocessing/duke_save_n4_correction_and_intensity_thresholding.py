import os
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
import SimpleITK as sitk

def get_n4_corrected_slice(pre_slice_arr, post_slice_arr):
    pre_sitk = sitk.GetImageFromArray(pre_slice_arr.astype(np.float32))
    post_sitk = sitk.GetImageFromArray(post_slice_arr.astype(np.float32))
    
    #Create intensity thresholding mask from pre-contrast to identify tissue 
    mask = sitk.OtsuThreshold(pre_sitk, 0, 1, 200)
    
    #Fit N4 corrector to pre-contrast
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    try:
        _ = corrector.Execute(pre_sitk, mask)
        log_bias_field = corrector.GetLogBiasFieldAsImage(pre_sitk)
        corrected_post_sitk = post_sitk / sitk.Exp(log_bias_field)
        return sitk.GetArrayFromImage(corrected_post_sitk)
    except Exception:
        return post_slice_arr

def extract_3_slice_stack(patient_list, root_dir, anno_path, output_dir):
    annotations = pd.read_excel(anno_path)
    pre_terms = ['pre', '000', 'baseline']
    post_terms = ['1st pass', 'ph1', 'post_1', 'dyn1']
    post_terms2 = ['1st ax dyn']

    for patient_id in patient_list:
        print(f"--- Processing Patient: {patient_id} ---")
        anno = annotations[annotations['PatientID'] == patient_id]
        if anno.empty: continue
        
        #Original bounding box coordinates and slice range
        r1, r2, c1, c2, s1, s2 = (anno['StartRow'].iloc[0], anno['EndRow'].iloc[0], 
                                  anno['StartColumn'].iloc[0], anno['EndColumn'].iloc[0], 
                                  anno['StartSlice'].iloc[0], anno['EndSlice'].iloc[0])

        normalized_target = str(patient_id).replace('_', '').lower()
        duke_root = os.path.join(root_dir, "Duke-Breast-Cancer-MRI")
        
        try:
            #Folders 
            patient_folders = [f for f in os.listdir(duke_root) if f.replace('_', '').lower() == normalized_target]
            if not patient_folders: continue
            
            patient_base = os.path.join(duke_root, patient_folders[0])
            study_sub = [f for f in os.listdir(patient_base) if os.path.isdir(os.path.join(patient_base, f))][0]
            study_path = os.path.join(patient_base, study_sub)
            series_folders = os.listdir(study_path)
            
            pre_path = post_path = post_path2 = None
            for s_folder in series_folders:
                folder_low = s_folder.lower()
                if any(x in folder_low for x in pre_terms) and not pre_path:
                    pre_path = os.path.join(study_path, s_folder)
                if any(x in folder_low for x in post_terms) and not post_path:
                    post_path = os.path.join(study_path, s_folder)
                if any(x in folder_low for x in post_terms2) and not post_path2:
                    post_path2 = os.path.join(study_path, s_folder)

            if not pre_path: continue

            def load_series(path):
                dcms = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path) if pydicom.misc.is_dicom(os.path.join(path, f))]
                dcms.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
                return dcms

            pre_slices = load_series(pre_path)
            mid_idx = int(s1 + (s2 - s1) // 2) - 1
            corrected_post = None
            last_post_error = None

            for candidate_path in (post_path, post_path2):
                if not candidate_path:
                    continue

                try:
                    post_slices = load_series(candidate_path)
                    if not post_slices:
                        raise ValueError(f"No DICOM slices found in {candidate_path}")

                    idx = max(0, min(mid_idx, len(pre_slices)-1, len(post_slices)-1))
                    
                    #N4 corrected post-contrast slice 
                    corrected_post = get_n4_corrected_slice(pre_slices[idx].pixel_array, post_slices[idx].pixel_array)
                    break
                except Exception as post_err:
                    last_post_error = post_err

            if corrected_post is None:
                if last_post_error:
                    raise last_post_error
                continue

            # 256x256 cropping 
            center_r = int((r1 + r2) // 2)
            center_c = int((c1 + c2) // 2)
            half_size = 128 # 256 / 2
            
            # Pad the image if the 256x256 crop goes out of bounds
            padded_arr = np.pad(corrected_post, pad_width=half_size, mode='constant', constant_values=0)
            
            #Adjust centers for padding
            adj_r = center_r + half_size
            adj_c = center_c + half_size
            
            # Extract exactly 256x256
            crop = padded_arr[adj_r - half_size : adj_r + half_size, 
                              adj_c - half_size : adj_c + half_size]

            # Normalize and Save
            vmin, vmax = np.percentile(crop, [1, 99])
            crop_norm = ((np.clip(crop, vmin, vmax) - vmin) / (vmax - vmin + 1e-5) * 255).astype(np.uint8)
            final_img = Image.fromarray(crop_norm)

            save_path = os.path.join(output_dir, str(patient_id))
            os.makedirs(save_path, exist_ok=True)
            
            for i in range(1, 4):
                final_img.save(os.path.join(save_path, f"slice_{i}_post_corrected_256.png"))

            print(f"  [SUCCESS] Saved 3 replicated 256x256 slices for {patient_id}.")

        except Exception as e:
            print(f"  [ERROR] {patient_id}: {e}")
            
PT_LIST = "C:/Users/y0qz1/Desktop/Extracted_Slices_20260329_errors_processed/PatientErrorsList2.xlsx"
ROOT ="C:/Users/y0qz1/Desktop/Extracted_Slices_20260329_errors/manifest-1777085455826"
MAP_FILE = "C:/Users/y0qz1/Desktop/Breast-Cancer-MRI-filepath_filename-mapping.xlsx"
ANNO_FILE = "C:/Users/y0qz1/Desktop/Annotation_Boxes.xlsx"
OUT = "C:/Users/y0qz1/Desktop/Extracted_Slices_20260329_errors_processed"


pts = pd.read_excel(PT_LIST)

patients = list(pts['PatientID'])



extract_3_slice_stack(patients, ROOT, ANNO_FILE, OUT)
