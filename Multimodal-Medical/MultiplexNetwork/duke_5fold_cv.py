import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
from tqdm import trange

from layers import AvgReadout, Attention, Discriminator, GCN
from utils import process


SCRIPT_DIR = Path(__file__).resolve().parent
MULTIMODAL_ROOT = SCRIPT_DIR.parent
HETMED_ROOT = MULTIMODAL_ROOT.parent

DEFAULT_CLINICAL_XLSX = HETMED_ROOT / "Data" / "Clinical_and_Other_Features_v6.xlsx"
DEFAULT_FEATURE_CSV = (
    MULTIMODAL_ROOT / "Image_embedder" / "DenseNet_121" / "breast_feature.csv"
)
DEFAULT_ID_CSV = MULTIMODAL_ROOT / "Image_embedder" / "DenseNet_121" / "breast_id.csv"

PATIENT_ID_CANDIDATES = ["Patient ID", "PatientID"]
LABEL_COLUMN = "Nottingham_Grade_v2"
DEFAULT_THRESHOLDS = "0.75,0.90,0.75,0.75"
DEFAULT_METAPATHS = "type0,type1,type2,type3"

# Cluster-based feature groups from duke_kmeans.py.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run 5-fold Duke DMGI cross-validation from breast_feature.csv and "
            "breast_id.csv without saving intermediate graph pickles."
        )
    )
    parser.add_argument("--clinical-xlsx", type=Path, default=DEFAULT_CLINICAL_XLSX)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--id-file", type=Path, default=DEFAULT_ID_CSV)
    parser.add_argument("--thresholds", type=str, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--embedder", default="DMGI")
    parser.add_argument("--metapaths", default=DEFAULT_METAPATHS)
    parser.add_argument("--nb_epochs", type=int, default=10000)
    parser.add_argument("--hid_units", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--l2_coef", type=float, default=0.0001)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--reg_coef", type=float, default=0.001)
    parser.add_argument("--sup_coef", type=float, default=0.01)
    parser.add_argument("--sc", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--nheads", type=int, default=1)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe-restarts", type=int, default=50)
    parser.add_argument("--probe-epochs", type=int, default=50)

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to save the 5-fold cross-validation summary JSON.",
    )
    parser.add_argument(
        "--saved-model-out",
        type=Path,
        default=None,
        help="Where to save the best fold model checkpoint.",
    )
    return parser.parse_args()


def resolve_output_paths(args):
    safe_metapaths = args.metapaths.replace(",", "_")
    if args.output_json is None:
        args.output_json = (
            SCRIPT_DIR
            / "intrinsic_metrics"
            / f"duke_5fold_cv_{args.embedder}_{safe_metapaths}.json"
        )
    if args.saved_model_out is None:
        args.saved_model_out = (
            SCRIPT_DIR
            / "saved_model"
            / f"best_duke_5fold_cv_{args.embedder}_{args.metapaths}.pkl"
        )


def resolve_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def load_embedding_inputs(feature_path, id_path):
    features = np.loadtxt(feature_path, delimiter=",", dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    ids = pd.read_csv(id_path, header=None)[0].astype(str).str.strip().tolist()
    if len(ids) != len(features):
        raise ValueError(
            f"Mismatch between ids ({len(ids)}) and image features ({len(features)})."
        )
    return ids, features


def build_duke_graph_base(feature_path, id_path, clinical_xlsx, thresholds):
    data = pd.read_excel(clinical_xlsx)
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
        raise ValueError(
            f"Duplicate patient ids found in clinical file: {duplicates[:10]}"
        )

    data[label_col] = data[label_col].apply(normalize_grade)
    data = data.dropna(subset=[label_col]).copy()
    if data.empty:
        raise ValueError(
            f"No rows remain after normalizing label column '{label_col}'."
        )

    ids, image_features = load_embedding_inputs(feature_path, id_path)
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

    ordered_image_features = np.asarray(ordered_image_features, dtype=np.float32)
    ordered_data = data.loc[ordered_ids].copy()

    adjacency_matrices = {}
    for group_index, group_columns in enumerate(resolved_groups):
        encoded_group = encode_feature_block(ordered_data, group_columns)
        group_values = minmax_scale(
            encoded_group.to_numpy(dtype=np.float32), axis=0, copy=True
        )
        cosine = cosine_similarity(group_values, group_values)
        adjacency = (cosine > thresholds[group_index]).astype(np.float32)
        adjacency_matrices[f"type{group_index}"] = adjacency

    all_non_image_columns = []
    for group_columns in resolved_groups:
        all_non_image_columns.extend(group_columns)
    encoded_non_image = encode_feature_block(ordered_data, all_non_image_columns)
    non_image_values = encoded_non_image.to_numpy(dtype=np.float32)
    concatenated_features = np.concatenate(
        (ordered_image_features, non_image_values), axis=1
    ).astype(np.float32)

    labels_int = ordered_data[label_col].astype(int).to_numpy()
    y = np.zeros((len(labels_int), 3), dtype=np.float32)
    for index, label in enumerate(labels_int):
        if label not in (1, 2, 3):
            raise ValueError(
                f"Unsupported label value '{label}' for patient '{ordered_ids[index]}'."
            )
        y[index, label - 1] = 1.0

    if missing_ids:
        print(
            f"Skipping {len(missing_ids)} embedded patients missing from the clinical workbook."
        )

    return {
        "patient_ids": ordered_ids,
        "label": y,
        "labels_int": labels_int.astype(np.int64),
        "feature": concatenated_features,
        "adjacency_matrices": adjacency_matrices,
    }


def prepare_static_tensors(graph_base, metapaths, sc, device):
    labels = torch.FloatTensor(graph_base["label"][np.newaxis]).to(device)

    feature_matrix = sp.lil_matrix(graph_base["feature"].astype(np.float32))
    dense_features = np.asarray(process.preprocess_features(feature_matrix), dtype=np.float32)
    features = [
        torch.FloatTensor(dense_features[np.newaxis]).to(device) for _ in metapaths
    ]

    adjacencies = []
    for metapath in metapaths:
        if metapath not in graph_base["adjacency_matrices"]:
            raise KeyError(
                f"Requested metapath '{metapath}' is not available in the constructed graph."
            )
        rownetwork = graph_base["adjacency_matrices"][metapath] + np.eye(
            graph_base["label"].shape[0], dtype=np.float32
        ) * sc
        normalized = process.normalize_adj(sp.csr_matrix(rownetwork))
        adj_tensor = process.sparse_mx_to_torch_sparse_tensor(normalized).to(device)
        adjacencies.append(adj_tensor)

    return features, adjacencies, labels


class ProbeLogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super().__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

    def forward(self, seq):
        return self.fc(seq)


class DMGIModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gcn = nn.ModuleList(
            [
                GCN(
                    args.ft_size,
                    args.hid_units,
                    args.activation,
                    args.drop_prob,
                    args.isBias,
                )
                for _ in range(args.nb_graphs)
            ]
        )
        self.disc = Discriminator(args.hid_units)
        self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units))
        self.readout_func = args.readout_func
        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])
        if args.isSemi:
            self.logistic = ProbeLogReg(args.hid_units, args.nb_classes).to(args.device)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.H)

    def forward(self, feature, adj, shuf, sparse):
        h_1_all = []
        h_2_all = []
        c_all = []
        logits = []
        result = {}
        for i in range(self.args.nb_graphs):
            h_1 = self.gcn[i](feature[i], adj[i], sparse)
            c = self.readout_func(h_1)
            c = self.args.readout_act_func(c)
            h_2 = self.gcn[i](shuf[i], adj[i], sparse)
            logit = self.disc(c, h_1, h_2, None, None)
            h_1_all.append(h_1)
            h_2_all.append(h_2)
            c_all.append(c)
            logits.append(logit)
        result["logits"] = logits

        if self.args.isAttn:
            h_1_all_lst = []
            h_2_all_lst = []
            for h_idx in range(self.args.nheads):
                h_1_all_, h_2_all_, _, _ = self.attn[h_idx](h_1_all, h_2_all, c_all)
                h_1_all_lst.append(h_1_all_)
                h_2_all_lst.append(h_2_all_)
            h_1_all = torch.mean(torch.cat(h_1_all_lst, 0), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all_lst, 0), 0).unsqueeze(0)
        else:
            h_1_all = torch.mean(torch.cat(h_1_all), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all), 0).unsqueeze(0)

        pos_reg_loss = ((self.H - h_1_all) ** 2).sum()
        neg_reg_loss = ((self.H - h_2_all) ** 2).sum()
        result["reg_loss"] = pos_reg_loss - neg_reg_loss

        if self.args.isSemi:
            result["semi"] = self.logistic(self.H).squeeze(0)
        return result


def class_distribution(one_hot_labels, indices):
    class_ids = np.argmax(one_hot_labels[indices], axis=1) + 1
    counts = np.bincount(class_ids, minlength=4)[1:]
    total = int(counts.sum())
    ratios = (counts / total).tolist() if total else [0.0, 0.0, 0.0]
    return {
        "size": total,
        "counts": {
            str(class_index + 1): int(count)
            for class_index, count in enumerate(counts)
        },
        "ratios": {
            str(class_index + 1): float(ratio)
            for class_index, ratio in enumerate(ratios)
        },
    }


def build_balanced_grade_folds(labels_int, n_splits, seed):
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    rng = np.random.default_rng(seed)
    labels_int = np.asarray(labels_int, dtype=np.int64)
    all_indices = np.arange(len(labels_int), dtype=np.int64)
    unique_labels = sorted(np.unique(labels_int))
    per_grade_test_chunks = {}

    for label in unique_labels:
        label_indices = all_indices[labels_int == label].copy()
        if len(label_indices) < n_splits:
            raise ValueError(
                f"Grade {label} has only {len(label_indices)} samples, fewer than n_splits={n_splits}."
            )
        rng.shuffle(label_indices)
        per_grade_test_chunks[int(label)] = [
            np.asarray(chunk, dtype=np.int64)
            for chunk in np.array_split(label_indices, n_splits)
        ]

    folds = []
    for fold_index in range(n_splits):
        test_parts = [
            per_grade_test_chunks[label][fold_index] for label in unique_labels
        ]
        test_idx = np.sort(np.concatenate(test_parts)).astype(np.int64)
        train_mask = np.ones(len(labels_int), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_indices[train_mask]
        folds.append((train_idx, test_idx))
    return folds


def evaluate_logreg_no_val(
    embeds,
    idx_train,
    idx_test,
    labels,
    device,
    num_restarts=50,
    num_epochs=50,
):
    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[0, idx_train]
    test_embs = embeds[0, idx_test]
    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    for restart in range(num_restarts):
        torch.manual_seed(restart)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(restart)
        log = ProbeLogReg(hid_units, nb_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        train_losses = []
        test_accs = []
        test_micro_f1s = []
        test_macro_f1s = []
        for _ in range(num_epochs):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

            log.eval()
            with torch.no_grad():
                logits = log(test_embs)
                preds = torch.argmax(logits, dim=1)
                test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                test_f1_macro = f1_score(
                    test_lbls.cpu(), preds.cpu(), average="macro"
                )
                test_f1_micro = f1_score(
                    test_lbls.cpu(), preds.cpu(), average="micro"
                )
            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

        best_epoch = int(np.argmin(train_losses))
        accs.append(test_accs[best_epoch])
        macro_f1s.append(test_macro_f1s[best_epoch])
        micro_f1s.append(test_micro_f1s[best_epoch])

    return {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "micro_f1_mean": float(np.mean(micro_f1s)),
        "micro_f1_std": float(np.std(micro_f1s)),
        "macro_f1_max": float(np.max(macro_f1s)),
        "micro_f1_max": float(np.max(micro_f1s)),
        "test_accuracy_max": float(np.max(accs)),
        "val_macro_f1_mean": None,
        "selection_strategy": "train_loss",
        "num_logreg_restarts": int(num_restarts),
        "num_logreg_epochs": int(num_epochs),
    }


def copy_state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def evaluate_fold_score(metrics):
    return (metrics["macro_f1_mean"], metrics["test_accuracy_max"])


def run_fold(
    fold_index,
    train_idx,
    test_idx,
    features,
    adjacencies,
    labels,
    args,
):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    train_idx_t = torch.LongTensor(train_idx).to(args.device)
    test_idx_t = torch.LongTensor(test_idx).to(args.device)
    train_lbls = torch.argmax(labels[0, train_idx_t], dim=1)

    model = DMGIModel(args).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2_coef
    )
    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()

    cnt_wait = 0
    best_training_loss = float("inf")
    best_state = None
    loss_history = []

    for _ in trange(args.nb_epochs, desc=f"Fold {fold_index + 1}", leave=False):
        model.train()
        optimizer.zero_grad()
        idx = np.random.permutation(args.nb_nodes)
        shuf = [feature[:, idx, :] for feature in features]

        lbl_1 = torch.ones(args.batch_size, args.nb_nodes, device=args.device)
        lbl_2 = torch.zeros(args.batch_size, args.nb_nodes, device=args.device)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        result = model(features, adjacencies, shuf, args.sparse)
        logits = result["logits"]

        xent_loss = None
        for logit in logits:
            if xent_loss is None:
                xent_loss = b_xent(logit, lbl)
            else:
                xent_loss += b_xent(logit, lbl)

        loss = xent_loss + args.reg_coef * result["reg_loss"]
        if args.isSemi:
            semi_loss = xent(result["semi"][train_idx_t], train_lbls)
            loss += args.sup_coef * semi_loss

        loss_value = float(loss.item())
        loss_history.append(loss_value)

        if loss_value < best_training_loss:
            best_training_loss = loss_value
            cnt_wait = 0
            best_state = copy_state_dict_to_cpu(model.state_dict())
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            break

        loss.backward()
        optimizer.step()

    if best_state is None:
        raise RuntimeError(f"Fold {fold_index + 1} failed to produce a best state.")

    model.load_state_dict(best_state)
    model.eval()
    evaluation = evaluate_logreg_no_val(
        model.H.data.detach(),
        train_idx_t,
        test_idx_t,
        labels,
        args.device,
        num_restarts=args.probe_restarts,
        num_epochs=args.probe_epochs,
    )

    return {
        "fold_index": int(fold_index),
        "epochs_completed": int(len(loss_history)),
        "best_training_loss": float(best_training_loss),
        "final_training_loss": float(loss_history[-1]) if loss_history else None,
        "split": {
            "train_size": int(len(train_idx)),
            "val_size": 0,
            "test_size": int(len(test_idx)),
            "train_ratio": float(len(train_idx) / args.nb_nodes),
            "val_ratio": 0.0,
            "test_ratio": float(len(test_idx) / args.nb_nodes),
        },
        "evaluation": evaluation,
        "state_dict": best_state,
    }


def summarize_fold_metrics(fold_results):
    metric_keys = [
        "macro_f1_mean",
        "macro_f1_std",
        "micro_f1_mean",
        "micro_f1_std",
        "macro_f1_max",
        "micro_f1_max",
        "test_accuracy_max",
    ]
    averages = {}
    stds = {}
    for key in metric_keys:
        values = [fold["evaluation"][key] for fold in fold_results]
        averages[key] = float(np.mean(values))
        stds[key] = float(np.std(values))
    return averages, stds


def main():
    args = parse_args()
    resolve_output_paths(args)
    args.device = resolve_device(args.device)
    args.batch_size = 1
    args.sparse = True
    args.isSemi = True
    args.isBias = False
    args.isAttn = True
    args.metapaths_list = args.metapaths.split(",")

    thresholds = [float(value) for value in args.thresholds.split(",")]
    if len(thresholds) != len(FEATURE_GROUPS):
        raise ValueError(
            f"Expected {len(FEATURE_GROUPS)} thresholds, got {len(thresholds)}."
        )
    if args.folds != 5:
        raise ValueError("This script is intended for 5-fold cross validation; set --folds 5.")
    if len(args.metapaths_list) != len(FEATURE_GROUPS):
        raise ValueError(
            f"Expected {len(FEATURE_GROUPS)} metapaths, got {len(args.metapaths_list)}."
        )

    set_seed(args.seed)
    graph_base = build_duke_graph_base(
        feature_path=args.feature_file,
        id_path=args.id_file,
        clinical_xlsx=args.clinical_xlsx,
        thresholds=thresholds,
    )

    args.nb_nodes = graph_base["label"].shape[0]
    args.ft_size = int(graph_base["feature"].shape[1])
    args.nb_classes = int(graph_base["label"].shape[1])
    args.nb_graphs = len(args.metapaths_list)
    args.readout_func = AvgReadout()
    args.readout_act_func = nn.Sigmoid()

    features, adjacencies, labels = prepare_static_tensors(
        graph_base, args.metapaths_list, args.sc, args.device
    )

    fold_splits = build_balanced_grade_folds(
        graph_base["labels_int"], n_splits=args.folds, seed=args.seed
    )
    fold_results = []
    best_fold_payload = None

    for fold_index, (train_idx, test_idx) in enumerate(fold_splits):
        set_seed(args.seed + fold_index)
        fold_payload = run_fold(
            fold_index=fold_index,
            train_idx=train_idx,
            test_idx=test_idx,
            features=features,
            adjacencies=adjacencies,
            labels=labels,
            args=args,
        )
        fold_results.append(fold_payload)

        score = evaluate_fold_score(fold_payload["evaluation"])
        if best_fold_payload is None or score > evaluate_fold_score(
            best_fold_payload["evaluation"]
        ):
            best_fold_payload = fold_payload

        print(
            "Fold {}/{}: Macro-F1 mean {:.4f}, Micro-F1 mean {:.4f}, Test accuracy max {:.4f}".format(
                fold_index + 1,
                args.folds,
                fold_payload["evaluation"]["macro_f1_mean"],
                fold_payload["evaluation"]["micro_f1_mean"],
                fold_payload["evaluation"]["test_accuracy_max"],
            )
        )

    if best_fold_payload is None:
        raise RuntimeError("No fold results were produced.")

    average_metrics, std_across_folds = summarize_fold_metrics(fold_results)

    args.saved_model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_fold_payload["state_dict"], args.saved_model_out)

    labels_np = graph_base["label"]
    per_fold_json = []
    for fold_payload in fold_results:
        fold_train_idx, fold_test_idx = fold_splits[fold_payload["fold_index"]]
        per_fold_json.append(
            {
                "fold_index": fold_payload["fold_index"],
                "epochs_completed": fold_payload["epochs_completed"],
                "best_training_loss": fold_payload["best_training_loss"],
                "final_training_loss": fold_payload["final_training_loss"],
                "split": fold_payload["split"],
                "class_weighting": {
                    "train": class_distribution(labels_np, fold_train_idx),
                    "test": class_distribution(labels_np, fold_test_idx),
                },
                "evaluation": fold_payload["evaluation"],
            }
        )

    summary = {
        "description": (
            "5-fold Duke DMGI cross-validation built directly from breast_feature.csv, "
            "breast_id.csv, and the clinical workbook without saving intermediate graph pickles."
        ),
        "split_strategy": "grade_balanced",
        "feature_file": str(args.feature_file.resolve()),
        "id_file": str(args.id_file.resolve()),
        "clinical_xlsx": str(args.clinical_xlsx.resolve()),
        "saved_model_out": str(args.saved_model_out.resolve()),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "thresholds": thresholds,
        "metapaths": args.metapaths_list,
        "hyperparameters": {
            "embedder": args.embedder,
            "lr": float(args.lr),
            "hid_units": int(args.hid_units),
            "reg_coef": float(args.reg_coef),
            "sup_coef": float(args.sup_coef),
            "l2_coef": float(args.l2_coef),
            "drop_prob": float(args.drop_prob),
            "sc": float(args.sc),
            "patience": int(args.patience),
            "nb_epochs": int(args.nb_epochs),
            "nheads": int(args.nheads),
            "activation": args.activation,
            "device": str(args.device),
            "probe_restarts": int(args.probe_restarts),
            "probe_epochs": int(args.probe_epochs),
        },
        "graph": {
            "num_patients": int(args.nb_nodes),
            "feature_dim": int(args.ft_size),
            "overall_class_weighting": class_distribution(
                labels_np, np.arange(args.nb_nodes)
            ),
        },
        "average_test_metrics_across_folds": average_metrics,
        "std_test_metrics_across_folds": std_across_folds,
        "best_fold": {
            "fold_index": int(best_fold_payload["fold_index"]),
            "selection_metric": "macro_f1_mean_then_test_accuracy_max",
            "evaluation": best_fold_payload["evaluation"],
            "split": best_fold_payload["split"],
            "class_weighting": {
                "train": class_distribution(
                    labels_np, fold_splits[best_fold_payload["fold_index"]][0]
                ),
                "test": class_distribution(
                    labels_np, fold_splits[best_fold_payload["fold_index"]][1]
                ),
            },
            "epochs_completed": int(best_fold_payload["epochs_completed"]),
            "best_training_loss": float(best_fold_payload["best_training_loss"]),
        },
        "per_fold": per_fold_json,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print()
    print("Average over {} folds:".format(args.folds))
    print(
        "  Macro-F1 mean {:.4f} | Micro-F1 mean {:.4f} | Test accuracy max {:.4f}".format(
            summary["average_test_metrics_across_folds"]["macro_f1_mean"],
            summary["average_test_metrics_across_folds"]["micro_f1_mean"],
            summary["average_test_metrics_across_folds"]["test_accuracy_max"],
        )
    )
    print(f"Saved summary JSON: {args.output_json}")
    print(f"Saved best fold model: {args.saved_model_out}")


if __name__ == "__main__":
    main()
