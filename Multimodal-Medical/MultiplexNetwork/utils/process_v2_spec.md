# Process v2 Functional Specification

This document defines the behavior implemented by `process_v2.py`.

## Purpose

Provide data-loading, sparse-matrix preprocessing, graph normalization, and tensor conversion helpers for DMGI-style training.

## Public API

The module must provide these functions:

- `loads(args)`
- `parse_skipgram(fname)`
- `accuracy(output, labels)`
- `adj_to_bias(adj, sizes, nhood=1)`
- `sample_mask(idx, l)`
- `sparse_to_tuple(sparse_mx, insert_batch=False)`
- `preprocess_features(features)`
- `normalize_adj(adj)`
- `preprocess_adj(adj)`
- `sparse_mx_to_torch_sparse_tensor(sparse_mx)`

## Dataset Loading

`loads(args)` must:

1. Read `data/<dataset>.pkl`, where `<dataset>` is `args.dataset`.
2. Use `args.metapaths_list` as the list of graph keys to extract.
3. Use `args.sc` as a self-connection scale added to each graph's identity matrix.
4. Return:
   - a list of CSR adjacency matrices, one per metapath;
   - a list of feature matrices, repeated once per graph;
   - the label matrix;
   - flattened train indices;
   - flattened validation indices;
   - flattened test indices.
5. Convert the feature matrix to a SciPy LIL sparse matrix before repeating it.

The pickle is expected to contain:

- `label`
- `feature`
- `train_idx`
- `val_idx`
- `test_idx`
- one adjacency matrix per requested metapath key.

## Matrix Helpers

- `preprocess_features` row-normalizes a sparse feature matrix and returns a dense matrix.
- `normalize_adj` performs symmetric adjacency normalization, returning a COO sparse matrix.
- `preprocess_adj` adds an identity matrix before adjacency normalization and returns tuple-format sparse data.
- `sparse_to_tuple` converts one sparse matrix or a list of sparse matrices to `(coords, values, shape)`, optionally adding a leading batch coordinate.
- `sparse_mx_to_torch_sparse_tensor` converts a SciPy sparse matrix to a PyTorch sparse float tensor.

## Other Helpers

- `parse_skipgram` parses skipgram text embeddings into a dense numpy matrix indexed by zero-based node IDs.
- `accuracy` returns the fraction of labels matched by the maximum-logit predictions.
- `adj_to_bias` expands graph neighborhoods and returns the large negative attention bias mask used by GAT-style models.
- `sample_mask` returns a boolean numpy mask with selected indices set to true.
