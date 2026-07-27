# HetMed_Experiments

This is an independent experimental implementation inspired by the HetMed and
DMGI papers. It is not affiliated with or endorsed by their authors and does
not contain source code from their unlicensed reference implementations.

The repository provides a project implementation based on Pytorch and PyTorch Geometric 
including experiments inspired by the Deep Multiplex Graph Infomax strategy described
in “Heterogeneous Graph Learning for Multi-modal Medical Data Analysis” (Kim
et al., 2023). The experiments focus on the
[Duke-Breast-Cancer-MRI collection](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)
available from The Cancer Imaging Archive (TCIA).

## Experimental scope

This project explores the following choices:

1. **Alternative MRI preprocessing.** A pre-contrast image is used to create
   an intensity-thresholding mask and to estimate an
   N4 bias field. The estimated field is then applied to the corresponding
   post-contrast image. After cropping and intensity normalization, the
   resulting grayscale image is converted to RGB, duplicating its grayscale
   intensities across the three input channels expected by the image encoder.
   Separately, the current preprocessing script saves three copies of the
   selected post-contrast slice for each patient.

2. **RadImageNet initialization.** As an alternative to the domain-adaptive
   pretraining evaluated in the original HetMed methodology, these experiments utilize a
   RadImageNet-pretrained DenseNet-121 encoder for generating image
   embeddings. RadImageNet weights are obtained separately and are not
   distributed by this repository.

3. **Nottingham grade preparation.** The experimental clinical input can use
   a Nottingham grade derived from the available tubular-formation, nuclear-
   pleomorphism, and mitotic-count scores: totals of 3–5, 6–7, and 8–9
   correspond to grades 1, 2, and 3, respectively. This follows the grade
   calculation reported by Hadidchi et al. in
   [“A deep learning framework to stratify Nottingham histologic grade 2
   breast tumors based on dynamic contrast-enhanced
   MRI”](https://doi.org/10.1007/s00330-025-12208-6) (European Radiology, 2026).
   The repository expects this derived value in the `Nottingham_Grade_v2` input column; patients with grade that remains
   unavailable are excluded.

5. **Preassigned non-image feature views.** The graph-construction code does
   not run a clustering algorithm to select non-image features. Instead,
   features are assigned to predefined groups, and each group defines a
   patient-similarity graph view using thresholded cosine similarity. These
   manually specified experimental feature groups should not be interpreted
   as validated clinical groupings.

6. **Grade-aware data splits.** Training, validation, and test indices are
   allocated by grade. By default, the split allocation follows the overall
   grade proportions. When the optional balanced-training mode is enabled,
   the training distribution may be adjusted while the validation and test
   distributions are constrained to remain within the code’s fixed maximum
   per-grade distribution-drift criterion relative to the full experimental
   dataset. 

The repository contains source code for preprocessing locally obtained data,
generating image embeddings, constructing multiplex patient graphs, and
training a PyTorch Geometric-based DMGI model. It does not distribute medical
data, clinical tables, annotations, embeddings, graph pickle files, pretrained
weights, or trained checkpoints.

See:

- [`Multimodal-Medical/README.md`](Multimodal-Medical/README.md) for the
  experiment overview.
- [`Multimodal-Medical/Preprocessing/README.md`](Multimodal-Medical/Preprocessing/README.md)
  for authorized data acquisition and local preprocessing.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance, licenses,
  research citations, and external-resource terms.
