import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale


SCRIPT_DIR = Path(__file__).resolve().parent
MULTIMODAL_ROOT = SCRIPT_DIR.parents[1]
HETMED_ROOT = SCRIPT_DIR.parents[2]

CLINICAL_XLSX = HETMED_ROOT / "Data" / "Clinical_and_Other_Features_v6.xlsx"
#CLINICAL_XLSX = HETMED_ROOT / "Data" / "Clinical_and_Other_Features_v6_test_matching_breast_ids.xlsx"
EMBEDDING_DIR = MULTIMODAL_ROOT / "Image_embedder" / "DenseNet_121"

#EMBEDDING_DIR = MULTIMODAL_ROOT/"Image_embedder"/"SimCLR"/"extracted_feature"
OUTPUT_PKL = MULTIMODAL_ROOT / "MultiplexNetwork" / "data" / "duke.pkl"

PATIENT_ID_CANDIDATES = ["Patient ID", "PatientID"]
LABEL_COLUMN = "Nottingham_Grade_v2"
MAX_HOLDOUT_DISTRIBUTION_DRIFT = 0.09



FEATURE_GROUPS = [
    [
        "Menopause_at_Dx",
        "ER",
        "PR",
        "Surgery",
        "Adjuvant_RT",
        "Adjuvant_Endocrine_Therapy_Medications",
        "Pec_Chest_Involvement",
    ],
    [
        "HER2",
        "Multicentric_Multifocal",
        "Lympadenopathy_Susp_Nodes",
        "Definitive_Surgery_Type",
        "Neoadjuvant_Chemotherapy",
        "Adjuvant_Chemotherapy",
        "Neoadjuvant_Anti_Her2_Neu_Therapy",
        "Adjuvant_Anti_Her2_Neu_Therapy",
    ],
    [
        "Mets_at_Presentation",
        "Contralateral_Involvement",
        "Staging_Mx",
        "Skin_Nipple_Involvement",
        "Neoadjuvant_RT",
        "Recurrence",
        "Known_Ovarian_Status",
        "Oophorectomy_as_Endocrine_Therapy",
        "Neoadjuvant_Endocrine_Therapy_Med",
    ],
    ["Staging_N"],
]

# FEATURE_GROUPS = [
#     # Group 1: Receptor Subtype & Patient Baseline Biology
#     # Focuses on the core molecular driver of the tumor and patient hormonal state.
#     [
#         "ER",                                  
#         "PR",                                   
#         "HER2",                                 
#         "Menopause_at_Dx",                      
#         "Known_Ovarian_Status",                 
#     ],
    
#     # Group 2: Local Aggressiveness, Infiltration, & Multi-focality
#     # Directly correlates with higher Nottingham grades (worse structural differentiation).
#     [
#         "Multicentric_Multifocal",             
#         "Lympadenopathy_Susp_Nodes",            
#         "Skin_Nipple_Involvement",              
#         "Pec_Chest_Involvement",               
#         "Staging_N",                            
#     ],
    
#     # Group 3: Systemic Spread & Disease Recurrence
#     # Captures the macro-level clinical outcomes of a highly aggressive, high-grade tumor.
#     [
#         "Mets_at_Presentation",                
#         "Contralateral_Involvement",            
#         "Staging_Mx",                           
#         "Recurrence",                          
#     ],
    
#     # Group 4: Treatment Intensity & Therapeutic Pathways
#     # High-grade, aggressive tumors trigger aggressive treatment plans (Neoadjuvant chemo/HER2 therapy).
#     [
#         "Neoadjuvant_Chemotherapy",             
#         "Adjuvant_Chemotherapy",                
#         "Neoadjuvant_Anti_Her2_Neu_Therapy",    
#         "Adjuvant_Anti_Her2_Neu_Therapy",       
#         "Neoadjuvant_RT",                       
#         "Adjuvant_RT",                          
#         "Surgery",                              
#         "Definitive_Surgery_Type",              
#         "Neoadjuvant_Endocrine_Therapy_Med",    
#         "Adjuvant_Endocrine_Therapy_Medications",
#         "Oophorectomy_as_Endocrine_Therapy",    
#     ]
# ]

def parse_args():
    parser = argparse.ArgumentParser(description="duke_kmeans")
    parser.add_argument("--K", type=int, default=len(FEATURE_GROUPS))
    parser.add_argument("--thres", type=str, default="0.8,0.8,0.8,0.8")
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=OUTPUT_PKL,
        help="Path to save the generated Duke multiplex graph pickle.",
    )
    parser.add_argument(
        "--balanced_training",
        action="store_true",
        help="Make the training split as balanced as possible across class labels.",
    )
    return parser.parse_known_args()


def canonicalize(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def resolve_column(columns, requested_name):
    requested_key = canonicalize(requested_name)
    for column in columns:
        if canonicalize(column) == requested_key:
            return column
    raise KeyError(f"Could not find column matching '{requested_name}'.")


def resolve_first_existing(columns, requested_names):
    for requested_name in requested_names:
        try:
            return resolve_column(columns, requested_name)
        except KeyError:
            continue
    raise KeyError(
        "Could not find any of these columns: {}".format(", ".join(requested_names))
    )


def normalize_grade(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    grade_map = {
        "1": 1,
        "2": 2,
        "3": 3,
        "low": 1,
        "intermediate": 2,
        "high": 3,
    }
    if text in grade_map:
        return grade_map[text]

    import re

    match = re.search(r"\b([123])\b", text)
    return int(match.group(1)) if match else None


def encode_feature_block(dataframe, columns):
    block = dataframe[columns].copy()
    block = block.astype("string").fillna("__MISSING__")
    block = pd.get_dummies(block, columns=columns)
    return block.astype(np.float32)


def load_embedding_inputs():
    # feature_path = EMBEDDING_DIR / "breast_feature_all.csv"
    # id_path = EMBEDDING_DIR / "breast_id_all.csv"
    
    feature_path = EMBEDDING_DIR / "train_feature.csv"
    id_path = EMBEDDING_DIR / "train_id.csv"

    features = np.loadtxt(feature_path, delimiter=",", dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    ids = pd.read_csv(id_path, header=None)[0].astype(str).str.strip().tolist()
    if len(ids) != len(features):
        raise ValueError(
            f"Mismatch between ids ({len(ids)}) and image features ({len(features)})."
        )
    return ids, features


def allocate_equal_counts(label_counts, total_size, reserved_counts=None):
    if not label_counts:
        return {}

    if reserved_counts is None:
        reserved_counts = {label: 0 for label in label_counts}

    available_counts = {
        label: max(0, count - reserved_counts.get(label, 0))
        for label, count in label_counts.items()
    }
    count_per_label = min(
        min(available_counts.values()),
        total_size // len(available_counts),
    )
    return {label: count_per_label for label in label_counts}


def allocate_proportional_counts(
    label_counts,
    total_size,
    target_label_counts=None,
    minimum_counts=None,
):
    if target_label_counts is None:
        target_label_counts = label_counts
    if minimum_counts is None:
        minimum_counts = {label: 0 for label in label_counts}

    total_count = sum(target_label_counts.values())
    if total_count == 0:
        return {label: 0 for label in label_counts}

    selected_counts = {}
    for label, count in label_counts.items():
        selected_counts[label] = min(
            count,
            max(
                minimum_counts.get(label, 0),
                int(total_size * target_label_counts[label] / total_count),
            ),
        )

    while sum(selected_counts.values()) < total_size:
        candidates = [
            label
            for label, count in label_counts.items()
            if selected_counts[label] < count
        ]
        if not candidates:
            break

        label = max(
            candidates,
            key=lambda candidate: (
                total_size * target_label_counts[candidate] / total_count
                - selected_counts[candidate],
                label_counts[candidate] - selected_counts[candidate],
                -candidate,
            ),
        )
        selected_counts[label] += 1

    return selected_counts


def allocate_split_minimum_counts(label_counts, total_size):
    minimum_counts = {label: 0 for label in label_counts}
    if total_size <= 0:
        return minimum_counts

    labels_with_patients = [
        label
        for label, count in sorted(label_counts.items())
        if count > 0
    ]
    if total_size < len(labels_with_patients):
        labels_with_patients = sorted(
            labels_with_patients,
            key=lambda label: label_counts[label],
            reverse=True,
        )[:total_size]

    for label in labels_with_patients:
        minimum_counts[label] = 1

    return minimum_counts


def training_balance_score(split_counts, split_sizes, label_counts):
    balanced_training_distribution = 1 / len(label_counts)
    return sum(
        (
            split_counts["train"][label] / split_sizes["train"]
            - balanced_training_distribution
        )
        ** 2
        for label in label_counts
    )


def holdout_splits_are_representative(split_counts, split_sizes, label_counts):
    total_count = sum(label_counts.values())
    original_distribution = {
        label: count / total_count
        for label, count in label_counts.items()
    }

    for split_name in ("valid", "test"):
        split_size = split_sizes[split_name]
        if split_size == 0:
            continue
        for label in label_counts:
            split_distribution = split_counts[split_name][label] / split_size
            if (
                abs(split_distribution - original_distribution[label])
                > MAX_HOLDOUT_DISTRIBUTION_DRIFT
            ):
                return False
    return True


def balance_training_counts(split_counts, split_sizes, label_counts):
    while True:
        current_score = training_balance_score(split_counts, split_sizes, label_counts)
        best_score = current_score
        best_counts = None

        for underrepresented_label in label_counts:
            for overrepresented_label in label_counts:
                if underrepresented_label == overrepresented_label:
                    continue
                if (
                    split_counts["train"][underrepresented_label]
                    >= split_counts["train"][overrepresented_label]
                ):
                    continue

                for split_name in ("valid", "test"):
                    if split_counts[split_name][underrepresented_label] <= 1:
                        continue
                    if split_counts["train"][overrepresented_label] <= 0:
                        continue

                    candidate_counts = {
                        name: counts.copy()
                        for name, counts in split_counts.items()
                    }
                    candidate_counts[split_name][underrepresented_label] -= 1
                    candidate_counts["train"][underrepresented_label] += 1
                    candidate_counts["train"][overrepresented_label] -= 1
                    candidate_counts[split_name][overrepresented_label] += 1

                    if not holdout_splits_are_representative(
                        candidate_counts,
                        split_sizes,
                        label_counts,
                    ):
                        continue

                    candidate_score = training_balance_score(
                        candidate_counts,
                        split_sizes,
                        label_counts,
                    )
                    if candidate_score < best_score:
                        best_score = candidate_score
                        best_counts = candidate_counts

        if best_counts is None:
            return split_counts

        split_counts = best_counts


def split_indexes(labels, train_size, valid_size, test_size, balanced_training=False):
    indexes_by_label = {}
    for index, label in enumerate(labels):
        indexes_by_label.setdefault(label, []).append(index)

    for label_indexes in indexes_by_label.values():
        random.shuffle(label_indexes)

    label_counts = {
        label: len(label_indexes)
        for label, label_indexes in sorted(indexes_by_label.items())
    }
    train_counts = allocate_proportional_counts(
        label_counts,
        train_size,
        target_label_counts=label_counts,
    )
    counts_after_train = {
        label: label_counts[label] - train_counts[label]
        for label in label_counts
    }
    valid_counts = allocate_proportional_counts(
        counts_after_train,
        valid_size,
        target_label_counts=label_counts,
    )
    counts_after_valid = {
        label: counts_after_train[label] - valid_counts[label]
        for label in label_counts
    }
    test_counts = allocate_proportional_counts(
        counts_after_valid,
        test_size,
        target_label_counts=label_counts,
    )
    if balanced_training:
        split_counts = balance_training_counts(
            {
                "train": train_counts,
                "valid": valid_counts,
                "test": test_counts,
            },
            {
                "train": train_size,
                "valid": valid_size,
                "test": test_size,
            },
            label_counts,
        )
        train_counts = split_counts["train"]
        valid_counts = split_counts["valid"]
        test_counts = split_counts["test"]

    train_index = []
    valid_index = []
    test_index = []
    for label, label_indexes in sorted(indexes_by_label.items()):
        train_count = train_counts[label]
        valid_count = valid_counts[label]
        test_count = test_counts[label]
        train_index.extend(label_indexes[:train_count])
        valid_index.extend(label_indexes[train_count : train_count + valid_count])
        test_index.extend(
            label_indexes[train_count + valid_count : train_count + valid_count + test_count]
        )

    random.shuffle(train_index)
    random.shuffle(valid_index)
    random.shuffle(test_index)
    return train_index, valid_index, test_index


def main():
    args, _ = parse_args()
    if args.K != len(FEATURE_GROUPS):
        raise ValueError(
            f"This script uses {len(FEATURE_GROUPS)} preassigned feature groups, so --K must be {len(FEATURE_GROUPS)}."
        )

    thresholds = [float(value) for value in args.thres.split(",")]
    if len(thresholds) != len(FEATURE_GROUPS):
        raise ValueError(
            f"Expected {len(FEATURE_GROUPS)} comma-separated thresholds, got {len(thresholds)}."
        )

    data = pd.read_excel(CLINICAL_XLSX)

    patient_id_col = resolve_first_existing(data.columns, PATIENT_ID_CANDIDATES)
    label_col = resolve_column(data.columns, LABEL_COLUMN)

    resolved_groups = []
    for group in FEATURE_GROUPS:
        resolved_groups.append([resolve_column(data.columns, name) for name in group])

    used_columns = [patient_id_col, label_col]
    for group in resolved_groups:
        used_columns.extend(group)

    data = data[used_columns].copy()
    data = data.dropna(subset=[patient_id_col]).copy()
    data[patient_id_col] = data[patient_id_col].astype(str).str.strip()
    data = data[data[patient_id_col] != ""].copy()

    duplicate_ids = data[patient_id_col].duplicated()
    if duplicate_ids.any():
        duplicates = data.loc[duplicate_ids, patient_id_col].tolist()
        raise ValueError(f"Duplicate patient ids found in clinical file: {duplicates[:10]}")

    data[label_col] = data[label_col].apply(normalize_grade)
    data = data.dropna(subset=[label_col]).copy()

    if data.empty:
        raise ValueError(
            f"No rows remain after normalizing label column '{label_col}'."
        )

    ids, image_features = load_embedding_inputs()

    data = data.set_index(patient_id_col, drop=False)
    ordered_ids = []
    ordered_image_features = []
    missing_ids = []

    for index, patient_id in enumerate(ids):
        if patient_id in data.index:
            ordered_ids.append(patient_id)
            ordered_image_features.append(image_features[index])
        else:
            missing_ids.append(patient_id)

    if not ordered_ids:
        raise ValueError(
            "No overlap between breast_id.csv and the clinical workbook patient ids."
        )

    if missing_ids:
        print(
            f"Skipping {len(missing_ids)} embedded patients that are missing from the clinical workbook."
        )

    ordered_image_features = np.asarray(ordered_image_features, dtype=np.float32)
    ordered_data = data.loc[ordered_ids].copy()

    adjacency_matrices = []
    for group_index, group_columns in enumerate(resolved_groups):
        encoded_group = encode_feature_block(ordered_data, group_columns)
        group_values = minmax_scale(encoded_group.to_numpy(dtype=np.float32), axis=0, copy=True)
        cosine = cosine_similarity(group_values, group_values)
        adjacency = (cosine > thresholds[group_index]).astype(np.float32)
        adjacency_matrices.append(adjacency)

    all_non_image_columns = []
    for group_columns in resolved_groups:
        all_non_image_columns.extend(group_columns)
    encoded_non_image = encode_feature_block(ordered_data, all_non_image_columns)
    non_image_values = encoded_non_image.to_numpy(dtype=np.float32)

    concatenated_features = np.concatenate(
        (ordered_image_features, non_image_values), axis=1
    )

    labels = ordered_data[label_col].astype(int).tolist()
    y = np.zeros((len(labels), 3), dtype=np.float32)
    for index, label in enumerate(labels):
        if label not in (1, 2, 3):
            raise ValueError(f"Unsupported label value '{label}' for patient '{ordered_ids[index]}'.")
        y[index, label - 1] = 1.0

    indexes = list(range(len(labels)))
    train_size = int(len(indexes) * 0.70)
    valid_size = int(len(indexes) * 0.10)
    test_size = len(indexes) - train_size - valid_size

    train_index, valid_index, test_index = split_indexes(
        labels,
        train_size,
        valid_size,
        test_size,
        balanced_training=args.balanced_training,
    )

    train_index = np.array(train_index)
    valid_index = np.array(valid_index)
    test_index = np.array(test_index)

    duke_multi = {
        "label": y,
        "train_idx": train_index,
        "val_idx": valid_index,
        "test_idx": test_index,
        "feature": concatenated_features,
    }
    for group_index, adjacency in enumerate(adjacency_matrices):
        duke_multi[f"type{group_index}"] = adjacency

    output_pkl = Path(args.output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as handle:
        pickle.dump(duke_multi, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
