"""Linear-probe evaluation for learned node embeddings."""

import numpy as np
import torch
from sklearn.metrics import f1_score

from models import LogReg


def _score(labels, predictions):
    truth = labels.detach().cpu().numpy()
    predicted = predictions.detach().cpu().numpy()
    return (
        float((truth == predicted).mean()),
        float(f1_score(truth, predicted, average="macro")),
        float(f1_score(truth, predicted, average="micro")),
    )


def evaluate(embeds, idx_train, idx_val, idx_test, labels, device, isTest=True):
    embeddings = embeds.squeeze(0)
    targets = labels.squeeze(0).argmax(dim=1)
    has_validation = idx_val.numel() > 0
    records = []

    for _ in range(50):
        classifier = LogReg(embeddings.size(1), labels.size(2)).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        best_key = None
        best_test = None
        for _epoch in range(50):
            classifier.train()
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                classifier(embeddings[idx_train]), targets[idx_train]
            )
            loss.backward()
            optimizer.step()

            classifier.eval()
            with torch.no_grad():
                test_scores = _score(
                    targets[idx_test], classifier(embeddings[idx_test]).argmax(dim=1)
                )
                if has_validation:
                    validation_scores = _score(
                        targets[idx_val], classifier(embeddings[idx_val]).argmax(dim=1)
                    )
                    selection_key = validation_scores[1]
                else:
                    selection_key = -float(loss)
                if best_key is None or selection_key > best_key:
                    best_key, best_test = selection_key, test_scores
        records.append(best_test)

    values = np.asarray(records)
    metrics = {
        "test_accuracy_max": float(values[:, 0].max()),
        "macro_f1_mean": float(values[:, 1].mean()),
        "macro_f1_std": float(values[:, 1].std()),
        "macro_f1_max": float(values[:, 1].max()),
        "micro_f1_mean": float(values[:, 2].mean()),
        "micro_f1_std": float(values[:, 2].std()),
        "micro_f1_max": float(values[:, 2].max()),
        "selection_strategy": "validation_macro_f1" if has_validation else "train_loss",
        "num_logreg_restarts": 50,
        "num_logreg_epochs": 50,
    }
    if isTest:
        print(
            f"\t[Classification] Macro-F1: {metrics['macro_f1_mean']:.4f} "
            f"({metrics['macro_f1_std']:.4f}) | Micro-F1: "
            f"{metrics['micro_f1_mean']:.4f} ({metrics['micro_f1_std']:.4f})"
        )
        return metrics
    return None, metrics["macro_f1_mean"]
