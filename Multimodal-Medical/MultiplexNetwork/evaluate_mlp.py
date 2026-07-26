import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR / "saved_model" / "best_duke_DMGI_type0,type1,type2,type3.pkl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone MLP evaluation for a saved DMGI checkpoint using the same "
            "train/val/test split stored in the matching graph pickle."
        )
    )
    parser.add_argument(
        "--saved-model",
        "--checkpoint",
        dest="saved_model",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to a saved DMGI checkpoint. The checkpoint must contain the "
            "'H' embedding tensor."
        ),
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help=(
            "Optional graph pickle containing label/train/val/test split data. "
            "If omitted, the script infers data/<dataset>.pkl from the checkpoint name."
        ),
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=64,
        help="Hidden size for the MLP classifier.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability inside the MLP classifier.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the MLP classifier.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for the MLP classifier optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs per restart.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=50,
        help="Number of random restarts, matching the evaluation style in evaluate.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device string such as 'cuda:0' or 'cpu'. "
            "Defaults to CUDA when available."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the final evaluation metrics as JSON.",
    )
    return parser.parse_args()


def resolve_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_dataset_name_from_checkpoint(checkpoint_path):
    stem = checkpoint_path.stem
    if not stem.startswith("best_") or "_DMGI_" not in stem:
        raise ValueError(
            "Could not infer dataset name from checkpoint filename "
            f"'{checkpoint_path.name}'. Pass --graph explicitly."
        )
    return stem[len("best_") :].split("_DMGI_", 1)[0]


def resolve_graph_path(args):
    if args.graph is not None:
        return args.graph
    dataset = infer_dataset_name_from_checkpoint(args.saved_model)
    return SCRIPT_DIR / "data" / f"{dataset}.pkl"


def load_checkpoint_embeddings(checkpoint_path):
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "H" not in state_dict:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain the 'H' embedding tensor."
        )

    embeddings = state_dict["H"]
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise ValueError(
            f"Expected H to have shape (1, N, D), got {tuple(embeddings.shape)}."
        )

    return embeddings.squeeze(0).cpu().to(torch.float32)


def load_graph(graph_path):
    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)

    required_keys = {"label", "train_idx", "val_idx", "test_idx"}
    missing = required_keys.difference(graph.keys())
    if missing:
        raise KeyError(f"Graph pickle at {graph_path} is missing keys: {sorted(missing)}")

    labels = np.asarray(graph["label"])
    if labels.ndim != 2:
        raise ValueError(f"Expected labels to have shape (N, C), got {labels.shape}.")

    return {
        "labels": torch.from_numpy(labels).to(torch.float32),
        "idx_train": np.asarray(graph["train_idx"]).ravel(),
        "idx_val": np.asarray(graph["val_idx"]).ravel(),
        "idx_test": np.asarray(graph["test_idx"]).ravel(),
    }


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(self, x):
        return self.net(x)


def evaluate_single_restart(
    embeddings,
    labels,
    idx_train,
    idx_val,
    idx_test,
    device,
    hidden_units,
    dropout,
    lr,
    weight_decay,
    epochs,
):
    if len(idx_train) == 0:
        raise ValueError("The graph split has an empty train_idx; MLP evaluation cannot run.")
    if len(idx_test) == 0:
        raise ValueError("The graph split has an empty test_idx; MLP evaluation cannot run.")

    has_validation = len(idx_val) > 0
    train_embs = embeddings[idx_train].to(device)
    test_embs = embeddings[idx_test].to(device)

    train_lbls = torch.argmax(labels[idx_train], dim=1).to(device)
    test_lbls = torch.argmax(labels[idx_test], dim=1).to(device)
    if has_validation:
        val_embs = embeddings[idx_val].to(device)
        val_lbls = torch.argmax(labels[idx_val], dim=1).to(device)
    else:
        val_embs = None
        val_lbls = None

    model = MLPClassifier(
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_units,
        output_dim=labels.shape[1],
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []
    test_accs = []
    val_micro_f1s = []
    test_micro_f1s = []
    val_macro_f1s = []
    test_macro_f1s = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_embs)
        loss = criterion(logits, train_lbls)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            if has_validation:
                val_logits = model(val_embs)
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = torch.sum(val_preds == val_lbls).float() / val_lbls.shape[0]
                val_f1_macro = f1_score(
                    val_lbls.cpu(), val_preds.cpu(), average="macro"
                )
                val_f1_micro = f1_score(
                    val_lbls.cpu(), val_preds.cpu(), average="micro"
                )

            test_logits = model(test_embs)
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = torch.sum(test_preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(
                test_lbls.cpu(), test_preds.cpu(), average="macro"
            )
            test_f1_micro = f1_score(
                test_lbls.cpu(), test_preds.cpu(), average="micro"
            )

        if has_validation:
            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)
        test_accs.append(test_acc.item())
        test_macro_f1s.append(test_f1_macro)
        test_micro_f1s.append(test_f1_micro)

    if has_validation:
        best_acc_epoch = int(np.argmax(val_accs))
        best_macro_epoch = int(np.argmax(val_macro_f1s))
        best_micro_epoch = int(np.argmax(val_micro_f1s))
        best_val_macro_f1 = float(val_macro_f1s[best_macro_epoch])
        selection_strategy = "validation_metrics"
    else:
        best_epoch = int(np.argmin(train_losses))
        best_acc_epoch = best_epoch
        best_macro_epoch = best_epoch
        best_micro_epoch = best_epoch
        best_val_macro_f1 = None
        selection_strategy = "train_loss"

    return {
        "test_accuracy_at_best_val_accuracy": float(test_accs[best_acc_epoch]),
        "test_macro_f1_at_best_val_macro_f1": float(test_macro_f1s[best_macro_epoch]),
        "test_micro_f1_at_best_val_micro_f1": float(test_micro_f1s[best_micro_epoch]),
        "best_val_macro_f1": best_val_macro_f1,
        "selection_strategy": selection_strategy,
    }


def run_evaluation(args):
    device = resolve_device(args.device)
    embeddings = load_checkpoint_embeddings(args.saved_model)
    graph_path = resolve_graph_path(args)
    graph = load_graph(graph_path)

    labels = graph["labels"]
    idx_train = graph["idx_train"]
    idx_val = graph["idx_val"]
    idx_test = graph["idx_test"]

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            "Embedding count ({}) does not match label count ({}) from {}.".format(
                embeddings.shape[0], labels.shape[0], graph_path
            )
        )

    accs = []
    macro_f1s = []
    micro_f1s = []
    val_macro_f1s = []
    selection_strategy = None

    for restart in range(args.restarts):
        set_seed(args.seed + restart)
        restart_metrics = evaluate_single_restart(
            embeddings=embeddings,
            labels=labels,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            device=device,
            hidden_units=args.hidden_units,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
        )
        selection_strategy = restart_metrics["selection_strategy"]
        accs.append(restart_metrics["test_accuracy_at_best_val_accuracy"])
        macro_f1s.append(restart_metrics["test_macro_f1_at_best_val_macro_f1"])
        micro_f1s.append(restart_metrics["test_micro_f1_at_best_val_micro_f1"])
        if restart_metrics["best_val_macro_f1"] is not None:
            val_macro_f1s.append(restart_metrics["best_val_macro_f1"])

    return {
        "saved_model": str(args.saved_model.resolve()),
        "graph": str(graph_path.resolve()),
        "device": str(device),
        "selection_strategy": selection_strategy,
        "n_patients": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "hidden_units": int(args.hidden_units),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.epochs),
        "restarts": int(args.restarts),
        "split_sizes": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "micro_f1_mean": float(np.mean(micro_f1s)),
        "micro_f1_std": float(np.std(micro_f1s)),
        "macro_f1_max": float(np.max(macro_f1s)),
        "micro_f1_max": float(np.max(micro_f1s)),
        "test_accuracy_max": float(np.max(accs)),
        "val_macro_f1_mean": (
            float(np.mean(val_macro_f1s)) if val_macro_f1s else None
        ),
    }


def main():
    args = parse_args()
    metrics = run_evaluation(args)

    print(
        "\t[MLP Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(
            metrics["macro_f1_mean"],
            metrics["macro_f1_std"],
            metrics["micro_f1_mean"],
            metrics["micro_f1_std"],
        )
    )
    print(
        "\t[Maximums] Macro-F1: {:.4f} | Micro-F1: {:.4f} | Test accuracy: {:.4f}".format(
            metrics["macro_f1_max"],
            metrics["micro_f1_max"],
            metrics["test_accuracy_max"],
        )
    )
    print(
        "\t[Split Sizes] train={} val={} test={}".format(
            metrics["split_sizes"]["train"],
            metrics["split_sizes"]["val"],
            metrics["split_sizes"]["test"],
        )
    )
    print(f"\t[Selection] {metrics['selection_strategy']}")
    print(f"\t[Checkpoint] {metrics['saved_model']}")
    print(f"\t[Graph] {metrics['graph']}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
        print(f"\t[Metrics JSON] {args.output_json}")


if __name__ == "__main__":
    main()
