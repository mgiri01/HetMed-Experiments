# Third-party notices

## Incorporated or substantially adapted material

### PyTorch Geometric

Parts of the multiplex-network implementation, including the relation-specific graph convolutions, corruption procedure, summary computation, discriminator loss, and consensus regularization, are based on or substantially adapted from PyTorch Geometric’s DMGI example. The implementation also uses torch_geometric.nn.GCNConv.

- Project: <https://github.com/pyg-team/pytorch_geometric>
- License: MIT
- Preserved license text:
  `Multimodal-Medical/MultiplexNetwork/PYG_LICENSE`

Copyright (c) 2023 PyG Team <team@pyg.org>

## Research foundations

The following publications are cited as academic foundations. Their citation
does not imply that their reference source code is included or licensed here.

- S. Kim et al., "Heterogeneous Graph Learning for Multi-modal Medical Data
  Analysis," AAAI 2023.
- C. Park et al., "Unsupervised Attributed Multiplex Network Embedding," AAAI
  2020. <https://arxiv.org/abs/1911.06750>

This repository is not affiliated with, endorsed by, or an official
distribution of either publication's authors or reference implementation.

## External data

The repository does not distribute medical images, clinical tables,
annotations, patient identifiers, extracted embeddings, graph files, or
checkpoints. Duke Breast Cancer MRI data must be obtained directly from The
Cancer Imaging Archive and used under the terms stated by TCIA:

- Collection: <https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/>
- DOI: <https://doi.org/10.7937/TCIA.e3sv-re93>

At the time this notice was written, TCIA identified the collection's images,
clinical information, annotations, and mapping tables as CC BY-NC 4.0 and
specified a required citation. Users are responsible for reviewing the
current terms before downloading or processing the collection.

## External model weights

RadImageNet pretrained weights are not distributed in this repository. Users
who choose RadImageNet initialization must obtain the DenseNet-121 checkpoint
from the official project and comply with the terms and citation supplied
there:

- Project: <https://github.com/BMEII-AI/RadImageNet>
- Publication: <https://doi.org/10.1148/rai.210315>

Record the exact download source, upstream revision or release, filename, and
cryptographic checksum in experiment records. Do not assume that a source-code
license automatically covers a separately distributed weight file; verify the
terms attached to the exact checkpoint before use or redistribution.

## Runtime dependencies

This repository declares third-party runtime dependencies in
`requirements.txt`. They are installed separately and are not redistributed
by this repository. Each dependency remains governed by its own license:

- NumPy: BSD-3-Clause
- pandas: BSD-3-Clause
- SciPy: BSD-3-Clause
- scikit-learn: BSD-3-Clause
- NetworkX: BSD-3-Clause
- Matplotlib: PSF-based license
- tqdm: MPL-2.0 and MIT
- PyYAML: MIT
- openpyxl: MIT
- Pillow: HPND
- OpenCV: Apache-2.0
- pydicom: MIT
- SimpleITK: Apache-2.0
- PyTorch and torchvision: BSD-3-Clause
- PyTorch Geometric: MIT
- Transformers, Hugging Face Hub, and Accelerate: Apache-2.0
- SentencePiece: Apache-2.0


Package metadata and the corresponding upstream project should be checked
when creating a frozen environment because dependency licenses and versions
can change.
