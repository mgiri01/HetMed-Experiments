import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-patient DenseNet-121 image embeddings."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing patient subfolders or flat image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./moco"),
        help="Directory to save feature and id CSV files.",
    )
    parser.add_argument(
        "--weights",
        choices=["radimagenet", "imagenet", "none"],
        default="radimagenet",
        help=(
            "DenseNet-121 initialization. 'radimagenet' expects a local RadImageNet "
            "checkpoint. 'imagenet' may download torchvision weights on first use."
        ),
    )
    parser.add_argument(
        "--radimagenet-checkpoint",
        type=Path,
        default=Path(__file__).with_name("RadImageNet-DenseNet121.pt"),
        help=(
            "Path to the local PyTorch RadImageNet DenseNet-121 checkpoint. "
            "The official RadImageNet repo distributes pretrained PyTorch models separately."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for slice-level embedding extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. 'cuda:0' or 'cpu'. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--feature-filename",
        default="breast_feature.csv",
        help="Numeric feature CSV output name expected by the downstream Duke kmeans code.",
    )
    parser.add_argument(
        "--id-filename",
        default="breast_id.csv",
        help="Patient id CSV output name expected by the downstream Duke kmeans code.",
    )
    parser.add_argument(
        "--summary-filename",
        default="embedding_summary.csv",
        help="Optional summary CSV with patient ids and number of slices used.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def patient_sort_key(patient_id: str):
    prefix = "Breast_MRI_"
    if patient_id.startswith(prefix):
        suffix = patient_id[len(prefix) :]
        if suffix.isdigit():
            return (0, int(suffix))
    digits = "".join(ch for ch in patient_id if ch.isdigit())
    if digits:
        return (1, int(digits), patient_id)
    return (2, patient_id)


def discover_patient_images(root: Path) -> Dict[str, List[Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    patient_to_images: Dict[str, List[Path]] = {}
    direct_children = list(root.iterdir())
    subdirs = [path for path in direct_children if path.is_dir()]

    if subdirs:
        for patient_dir in sorted(subdirs):
            image_paths = sorted(
                [path for path in patient_dir.rglob("*") if is_image_file(path)]
            )
            if image_paths:
                patient_to_images[patient_dir.name] = image_paths
    else:
        image_paths = sorted([path for path in root.rglob("*") if is_image_file(path)])
        for image_path in image_paths:
            patient_id = infer_patient_id_from_filename(image_path.stem)
            patient_to_images.setdefault(patient_id, []).append(image_path)

    if not patient_to_images:
        raise RuntimeError(f"No image files found under {root}")

    return {
        patient_id: sorted(paths)
        for patient_id, paths in sorted(patient_to_images.items(), key=lambda item: patient_sort_key(item[0]))
    }


def infer_patient_id_from_filename(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    if "-" in stem:
        return stem.split("-")[0]
    return stem


class SliceDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), str(image_path)


def build_model(weights_name: str, device: torch.device):
    if weights_name == "radimagenet":
        model = models.densenet121(weights=None)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )
        weights = "radimagenet"
    elif weights_name == "imagenet":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        transform = weights.transforms()
    else:
        weights = None
        model = models.densenet121(weights=None)
        transform = models.DenseNet121_Weights.DEFAULT.transforms()

    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()
    return model, transform, weights


def load_state_dict_from_checkpoint(checkpoint_path: Path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {checkpoint_path}")
    return checkpoint


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        if new_key.startswith("backbone.0."):
            new_key = "features." + new_key[len("backbone.0.") :]
        elif new_key.startswith("backbone.1."):
            new_key = "classifier." + new_key[len("backbone.1.") :]
        elif new_key.startswith("backbone."):
            new_key = new_key[len("backbone.") :]
        normalized[new_key] = value
    return normalized


def load_radimagenet_weights(model: nn.Module, checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "RadImageNet checkpoint not found at "
            f"{checkpoint_path}. Download the official PyTorch RadImageNet "
            "weights and place them at this path or pass --radimagenet-checkpoint."
        )

    state_dict = load_state_dict_from_checkpoint(checkpoint_path)
    state_dict = normalize_state_dict_keys(state_dict)

    classifier_weight = state_dict.get("classifier.weight")
    classifier_bias = state_dict.get("classifier.bias")
    if classifier_weight is not None and classifier_bias is not None:
        out_features = classifier_weight.shape[0]
        if model.classifier.out_features != out_features:
            model.classifier = nn.Linear(model.classifier.in_features, out_features)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in missing if not key.startswith("classifier.")]
    if missing:
        raise RuntimeError(
            "Failed to load the RadImageNet DenseNet-121 backbone. "
            f"Missing non-classifier keys: {missing[:10]}"
        )
    return unexpected


def compute_slice_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    slice_features: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Embedding slices"):
            images = images.to(device, non_blocking=True)
            features = model(images).detach().cpu().numpy()
            for path_str, feature in zip(paths, features):
                slice_features[path_str] = feature
    return slice_features


def average_patient_embeddings(
    patient_to_images: Dict[str, List[Path]],
    slice_features: Dict[str, np.ndarray],
):
    patient_ids: List[str] = []
    features: List[np.ndarray] = []
    summary_rows = []

    for patient_id, image_paths in patient_to_images.items():
        patient_slice_features = [slice_features[str(path)] for path in image_paths]
        patient_feature = np.mean(np.stack(patient_slice_features, axis=0), axis=0)
        patient_ids.append(patient_id)
        features.append(patient_feature.astype(np.float32))
        summary_rows.append(
            {
                "PatientID": patient_id,
                "NumSlices": len(image_paths),
                "SourceDir": str(image_paths[0].parent),
            }
        )

    return patient_ids, np.stack(features, axis=0), pd.DataFrame(summary_rows)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    patient_to_images = discover_patient_images(args.input_dir)

    all_image_paths = []
    for image_paths in patient_to_images.values():
        all_image_paths.extend(image_paths)

    model, transform, weights = build_model(args.weights, device)
    if args.weights == "radimagenet":
        unexpected = load_radimagenet_weights(model, args.radimagenet_checkpoint)
        if unexpected:
            print(
                "Ignoring unexpected checkpoint keys: "
                + ", ".join(unexpected[:10])
            )
        model.classifier = nn.Identity()
        model = model.to(device)
        model.eval()
    dataset = SliceDataset(all_image_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Using device: {device}")
    print(f"Patients discovered: {len(patient_to_images)}")
    print(f"Slices discovered: {len(all_image_paths)}")
    if weights is None:
        print("DenseNet-121 initialized without pretrained weights.")
    elif weights == "radimagenet":
        print(
            "DenseNet-121 initialized with RadImageNet pretrained weights from "
            f"{args.radimagenet_checkpoint}."
        )
    else:
        print("DenseNet-121 initialized with ImageNet pretrained weights.")

    slice_features = compute_slice_embeddings(model, dataloader, device)
    patient_ids, patient_features, summary_df = average_patient_embeddings(
        patient_to_images, slice_features
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / args.feature_filename
    id_path = args.output_dir / args.id_filename
    summary_path = args.output_dir / args.summary_filename

    np.savetxt(feature_path, patient_features, delimiter=",")
    pd.Series(patient_ids).to_csv(id_path, header=False, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved patient embeddings to {feature_path}")
    print(f"Saved patient ids to {id_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Feature matrix shape: {patient_features.shape}")


if __name__ == "__main__":
    main()
