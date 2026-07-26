# Multiplex medical graph experiments

This repository contains independent experiments for learning patient
representations from image features and multiple clinical graph views. It does
not contain source code from the unlicensed HetMed or DMGI reference
implementations. The training implementation is independently integrated
around PyTorch Geometric and preserves PyG's MIT notice in
`MultiplexNetwork/PYG_LICENSE`.

The research design is informed by these publications:

- S. Kim et al., “Heterogeneous Graph Learning for Multi-modal Medical Data
  Analysis,” AAAI 2023.
- C. Park et al., “Unsupervised Attributed Multiplex Network Embedding,” AAAI
  2020.

Those citations acknowledge the underlying research ideas. This repository is
not affiliated with or endorsed by the papers' authors and is not an official
distribution of either paper's reference implementation.

## Training

Place a locally generated graph pickle in `MultiplexNetwork/data`, then run:

```bash
cd MultiplexNetwork
python main.py --dataset DATASET_NAME --metapaths type0,type1,type2,type3
```

Medical images, clinical tables, annotations, embeddings, graph pickle files,
and checkpoints must remain outside version control. See
`Preprocessing/README.md` for the authorized local-data workflow and the
repository-level `THIRD_PARTY_NOTICES.md` for provenance and licensing details.
