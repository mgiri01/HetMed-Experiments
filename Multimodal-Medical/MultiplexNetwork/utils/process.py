"""Dataset and sparse-matrix utilities used by multiplex experiments."""

from pathlib import Path
import pickle

import numpy as np
import scipy.sparse as sp
import torch


def loads(args):
    path = Path(__file__).resolve().parents[1] / "data" / f"{args.dataset}.pkl"
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    labels = np.asarray(payload["label"])
    identity = sp.eye(labels.shape[0], format="csr")
    graphs = [sp.csr_matrix(payload[name]) + args.sc * identity for name in args.metapaths_list]
    features = sp.csr_matrix(np.asarray(payload["feature"], dtype=np.float32))
    return (
        graphs,
        [features.copy() for _ in graphs],
        labels,
        np.asarray(payload["train_idx"]).ravel(),
        np.asarray(payload["val_idx"]).ravel(),
        np.asarray(payload["test_idx"]).ravel(),
    )


def preprocess_features(matrix):
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals != 0)
    return sp.diags(inverse) @ matrix


def normalize_adj(matrix):
    matrix = sp.coo_matrix(matrix, dtype=np.float32)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse_root = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    scale = sp.diags(inverse_root)
    return (scale @ matrix @ scale).tocoo()


def sparse_mx_to_torch_sparse_tensor(matrix):
    matrix = sp.coo_matrix(matrix, dtype=np.float32)
    indices = torch.tensor(np.vstack((matrix.row, matrix.col)), dtype=torch.long)
    values = torch.tensor(matrix.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def accuracy(output, labels):
    return output.argmax(dim=1).eq(labels).float().mean()
