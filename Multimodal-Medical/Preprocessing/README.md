# Local preprocessing workflow

The preprocessing programs convert locally obtained medical data into image
features and multiplex graph inputs. Dataset access, use, and citation terms
must be reviewed at the dataset provider before running them.

This repository does not provide medical images, clinical tables, annotations,
patient identifiers, feature exports, graph pickle files, or derived
checkpoints.

## Duke Breast Cancer MRI acquisition

The Duke-Breast-Cancer-MRI collection distributed by TCIA is de-identified.

1. Review the collection page and current license:
   <https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/>.
2. Cite the collection using the instructions on that page and its DOI:
   <https://doi.org/10.7937/TCIA.e3sv-re93>.
3. Download the authorized imaging, clinical, and annotation files directly
   from TCIA to a directory outside this repository.
4. Do not commit downloaded or derived patient data. The collection page
   currently identifies these materials as CC BY-NC 4.0 and specifies a
   required citation; users must confirm the current terms themselves.

The typical workflow is:

1. Download an authorized dataset to a location outside the repository.
2. Configure the relevant script with local input and output paths.
3. Generate image crops or embeddings from the de-identified dataset locally.
4. Generate a graph pickle containing `feature`, `label`, split indices, and
   one adjacency matrix for each requested metapath.
5. Place the graph input needed for a local run under
   `MultiplexNetwork/data`. 

The Duke-specific scripts use annotation ranges to select MRI slices and use
clinical feature groups to construct patient-similarity graph views. They do
not download source data automatically.
