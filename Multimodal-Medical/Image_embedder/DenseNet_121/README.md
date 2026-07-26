# DenseNet-121 Image Embeddings

This folder contains a standalone DenseNet-121 embedding generator that follows
the existing repository pattern:

- run a vision backbone on each image slice
- group slices by patient
- average slice embeddings into one feature vector per patient
- save separate feature and patient-id CSV outputs for downstream preprocessing scripts

## Expected Input Layout

The script supports two layouts:

1. Patient folders, which matches the Duke preprocessing outputs:

```text
Extracted_Slices_Validation_20260318/
  Breast_MRI_024/
    slice_1_post_corrected_256.png
    slice_2_post_corrected_256.png
    slice_3_post_corrected_256.png
```

2. Flat image files, where patient ids can be inferred from filenames.

## Example Usage

From this folder:

```bash
python generate_embeddings.py \
  --input-dir /path/outside/repository/duke-slices \
  --output-dir /path/outside/repository/duke-embeddings \
  --device cuda:0
```

This writes:

- `breast_feature.csv`: one numeric DenseNet-121 embedding row per patient
- `breast_id.csv`: patient ids in the same row order as `breast_feature.csv`
- `embedding_summary.csv`: patient id, number of slices used, and source path

These filenames match what the current Duke kmeans pipeline expects in
`Preprocessing/Non_image_preprocessing/duke_kmeans.py`.

## RadImageNet weights

RadImageNet weights are not bundled with this repository. Obtain the
DenseNet-121 checkpoint from the official RadImageNet project:
<https://github.com/BMEII-AI/RadImageNet>.

Before use, review the terms attached to the exact checkpoint and record its
download URL, upstream revision or release, filename, and cryptographic
checksum. A source repository's license should not be assumed to cover a
separately distributed weight file.

Pass the downloaded file explicitly:

```bash
python generate_embeddings.py \
  --input-dir /path/outside/repository/duke-slices \
  --output-dir /path/outside/repository/duke-embeddings \
  --weights radimagenet \
  --radimagenet-checkpoint /path/outside/repository/RadImageNet-DenseNet121.pt
```

## Notes

- DenseNet-121 produces 1024-dimensional embeddings after removing the final
  classifier layer.
- To avoid loading pretrained weights, add:

```bash
--weights none
```
