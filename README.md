# HetMed_Experiments

This is an independent experimental implementation inspired by the HetMed and
DMGI papers. It is not affiliated with or endorsed by their authors and does
not contain source code from their unlicensed reference implementations.

The repository contains source code for preprocessing locally obtained Duke
Breast Cancer MRI data, generating image embeddings, constructing multiplex
patient graphs, and training a PyTorch Geometric-based DMGI model. It does not
distribute medical data, clinical tables, annotations, embeddings, graph
pickle files, pretrained weights, or trained checkpoints.

See:

- [`Multimodal-Medical/README.md`](Multimodal-Medical/README.md) for the
  experiment overview.
- [`Multimodal-Medical/Preprocessing/README.md`](Multimodal-Medical/Preprocessing/README.md)
  for authorized data acquisition and local preprocessing.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance, licenses,
  research citations, and external-resource terms.

The top-level [`LICENSE`](LICENSE) applies only to original code and
documentation owned by this repository's contributors.
