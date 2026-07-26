"""Multiplex Deep Graph Infomax trainer built on PyTorch Geometric.

The relation-wise corruption and consensus objective follow the PyG DMGI
example distributed under the MIT License. The required notice is available in
``../PYG_LICENSE``. Project integration in this file is independently written.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import trange

from embedder import embedder
from evaluate import evaluate
from layers import Attention, Discriminator, GCN
from models import LogReg
from scheduler_utils import build_lr_scheduler, scheduler_config_to_dict


class MultiplexDMGI(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoders = nn.ModuleList(
            GCN(args.ft_size, args.hid_units, args.activation, args.drop_prob, args.isBias)
            for _ in range(args.nb_graphs)
        )
        self.discriminator = Discriminator(args.hid_units)
        self.Z = nn.Parameter(torch.empty(1, args.nb_nodes, args.hid_units))
        nn.init.xavier_uniform_(self.Z)
        self.attention = (
            nn.ModuleList(Attention(args) for _ in range(args.nheads))
            if args.isAttn else None
        )
        self.classifier = LogReg(args.hid_units, args.nb_classes) if args.isSemi else None

    @property
    def H(self):
        """Compatibility alias for checkpoints and evaluation utilities."""
        return self.Z

    def _consensus(self, positives, negatives, summaries):
        if self.attention is None:
            return torch.stack(positives).mean(dim=0), torch.stack(negatives).mean(dim=0)
        positive_heads, negative_heads = [], []
        for head in self.attention:
            positive, negative, _summary, _weights = head(positives, negatives, summaries)
            positive_heads.append(positive)
            negative_heads.append(negative)
        return torch.stack(positive_heads).mean(dim=0), torch.stack(negative_heads).mean(dim=0)

    def forward(self, features, adjacencies, sparse=True):
        positives, negatives, summaries, logits = [], [], [], []
        for encoder, feature, adjacency in zip(self.encoders, features, adjacencies):
            positive = encoder(feature, adjacency, sparse)
            negative_input = feature[:, torch.randperm(feature.size(1), device=feature.device)]
            negative = encoder(negative_input, adjacency, sparse)
            summary = positive.mean(dim=1).sigmoid()
            positives.append(positive)
            negatives.append(negative)
            summaries.append(summary)
            logits.append(self.discriminator(summary, positive, negative))
        positive_mean, negative_mean = self._consensus(positives, negatives, summaries)
        regularizer = (self.Z - positive_mean).square().sum() - (self.Z - negative_mean).square().sum()
        output = {
            "pos_hs": positives,
            "neg_hs": negatives,
            "summaries": summaries,
            "logits": logits,
            "reg_loss": regularizer,
        }
        if self.classifier is not None:
            output["semi"] = self.classifier(self.Z).squeeze(0)
        return output

    @staticmethod
    def contrastive_loss(output):
        total = output["logits"][0].new_zeros(())
        for scores in output["logits"]:
            nodes = scores.size(1) // 2
            target = torch.cat((scores.new_ones(scores.size(0), nodes), scores.new_zeros(scores.size(0), nodes)), dim=1)
            total = total + nn.functional.binary_cross_entropy_with_logits(scores, target)
        return total


# Historical public name retained for callers that import ``modeler``.
modeler = MultiplexDMGI


class DMGI(embedder):
    def __init__(self, args):
        super().__init__(args)

    def _paths(self):
        root = Path(__file__).resolve().parents[1]
        metrics = root / "intrinsic_metrics"
        checkpoints = root / "saved_model"
        metrics.mkdir(parents=True, exist_ok=True)
        checkpoints.mkdir(parents=True, exist_ok=True)
        name = getattr(self.args, "checkpoint_name", None)
        if not name:
            name = f"best_{self.args.dataset}_{self.args.embedder}_{self.args.metapaths}.pkl"
        elif not name.endswith(".pkl"):
            name += ".pkl"
        stem = f"{self.args.dataset}_{self.args.embedder}_{self.args.metapaths.replace(',', '_')}"
        return checkpoints / name, metrics / f"training_metrics_{stem}.json"

    def training(self):
        features = [value.to(self.args.device) for value in self.features]
        adjacencies = [value.to(self.args.device) for value in self.adj]
        network = MultiplexDMGI(self.args).to(self.args.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        scheduler = build_lr_scheduler(self.args, optimizer)
        checkpoint, metrics_path = self._paths()
        best, stale = float("inf"), 0
        curves = {"overall_objective": [], "bce_discriminator": [], "supervised": [], "consensus": []}

        for _ in trange(self.args.nb_epochs):
            network.train()
            optimizer.zero_grad()
            output = network(features, adjacencies, self.args.sparse)
            contrastive = network.contrastive_loss(output)
            consensus = self.args.reg_coef * output["reg_loss"]
            supervised = contrastive.new_zeros(())
            if self.args.isSemi:
                supervised = self.args.sup_coef * nn.functional.cross_entropy(
                    output["semi"][self.idx_train], self.train_lbls
                )
            objective = contrastive + consensus + supervised
            curves["overall_objective"].append(float(objective.detach()))
            curves["bce_discriminator"].append(float(contrastive.detach()))
            curves["supervised"].append(float(supervised.detach()))
            curves["consensus"].append(float(consensus.detach()))
            if curves["overall_objective"][-1] < best:
                best, stale = curves["overall_objective"][-1], 0
                torch.save(network.state_dict(), checkpoint)
            else:
                stale += 1
            if stale >= self.args.patience:
                break
            objective.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        network.load_state_dict(torch.load(checkpoint, map_location=self.args.device))
        network.eval()
        evaluation = evaluate(network.Z.detach(), self.idx_train, self.idx_val, self.idx_test, self.labels, self.args.device)
        report = {
            "dataset": self.args.dataset,
            "metapaths": self.args.metapaths.split(","),
            "epochs_completed": len(curves["overall_objective"]),
            "best_training_loss": best,
            "optimizer": {"name": "Adam", "lr": self.args.lr, "weight_decay": self.args.l2_coef},
            "lr_scheduler": scheduler_config_to_dict(self.args),
            "curves": curves,
            "checkpoint_path": str(checkpoint),
            "evaluation": evaluation,
        }
        metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True))
