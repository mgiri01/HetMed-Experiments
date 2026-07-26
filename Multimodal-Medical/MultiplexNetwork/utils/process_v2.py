from pathlib import Path
import pickle

import numpy as np
import scipy.sparse as sp
import torch


def loads(args):
    dataset_path = Path("data") / f"{args.dataset}.pkl"
    with open(dataset_path, "rb") as handle:
        data = pickle.load(handle)

    labels = data["label"]
    node_count = labels.shape[0]
    identity = np.eye(node_count)

    networks = []
    for metapath_name in args.metapaths_list:
        graph = data[metapath_name] + identity * args.sc
        networks.append(sp.csr_matrix(graph))

    feature_matrix = sp.lil_matrix(data["feature"].astype(float))
    feature_views = [feature_matrix for _ in networks]

    return (
        networks,
        feature_views,
        labels,
        data["train_idx"].ravel(),
        data["val_idx"].ravel(),
        data["test_idx"].ravel(),
    )


def parse_skipgram(fname):
    tokens = Path(fname).read_text().split()
    node_count = int(tokens[0])
    feature_count = int(tokens[1])
    embeddings = np.empty((node_count, feature_count))

    cursor = 2
    for _ in range(node_count):
        node_index = int(tokens[cursor]) - 1
        cursor += 1
        row_values = [float(value) for value in tokens[cursor : cursor + feature_count]]
        embeddings[node_index, :] = row_values
        cursor += feature_count

    return embeddings


def accuracy(output, labels):
    predicted_labels = output.max(dim=1)[1].type_as(labels)
    matches = predicted_labels.eq(labels).double()
    return matches.sum() / len(labels)


def adj_to_bias(adj, sizes, nhood=1):
    graph_count = adj.shape[0]
    reachable = np.empty(adj.shape)

    for graph_index in range(graph_count):
        node_count = adj.shape[1]
        reachable[graph_index] = np.eye(node_count)
        graph_with_self = adj[graph_index] + np.eye(node_count)
        for _ in range(nhood):
            reachable[graph_index] = np.matmul(reachable[graph_index], graph_with_self)

        active_size = sizes[graph_index]
        reachable[graph_index, :active_size, :active_size] = (
            reachable[graph_index, :active_size, :active_size] > 0.0
        ).astype(float)

    return -1e9 * (1.0 - reachable)


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.asarray(mask, dtype=np.bool_)


def _single_sparse_tuple(matrix, insert_batch):
    coo_matrix = matrix if sp.isspmatrix_coo(matrix) else matrix.tocoo()
    if insert_batch:
        coordinates = np.vstack(
            (
                np.zeros(coo_matrix.row.shape[0]),
                coo_matrix.row,
                coo_matrix.col,
            )
        ).transpose()
        matrix_shape = (1,) + coo_matrix.shape
    else:
        coordinates = np.vstack((coo_matrix.row, coo_matrix.col)).transpose()
        matrix_shape = coo_matrix.shape

    return coordinates, coo_matrix.data, matrix_shape


def sparse_to_tuple(sparse_mx, insert_batch=False):
    if isinstance(sparse_mx, list):
        return [
            _single_sparse_tuple(matrix, insert_batch)
            for matrix in sparse_mx
        ]
    return _single_sparse_tuple(sparse_mx, insert_batch)


def preprocess_features(features):
    row_sums = np.asarray(features.sum(axis=1)).flatten()
    inverse_rows = np.divide(
        1.0,
        row_sums,
        out=np.zeros_like(row_sums, dtype=float),
        where=row_sums != 0,
    )
    normalizer = sp.diags(inverse_rows)
    return normalizer.dot(features).todense()


def normalize_adj(adj):
    coo_adj = sp.coo_matrix(adj)
    degree = np.asarray(coo_adj.sum(axis=1)).flatten()
    inv_sqrt_degree = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree, dtype=float),
        where=degree != 0,
    )
    scale = sp.diags(inv_sqrt_degree)
    return coo_adj.dot(scale).transpose().dot(scale).tocoo()


def preprocess_adj(adj):
    with_self_loops = adj + sp.eye(adj.shape[0])
    return sparse_to_tuple(normalize_adj(with_self_loops))


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    coo_matrix = sparse_mx.tocoo().astype(np.float32)
    coordinates = np.vstack((coo_matrix.row, coo_matrix.col)).astype(np.int64)
    indices = torch.from_numpy(coordinates)
    values = torch.from_numpy(coo_matrix.data)
    return torch.sparse.FloatTensor(indices, values, torch.Size(coo_matrix.shape))
