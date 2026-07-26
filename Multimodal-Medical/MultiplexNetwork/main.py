"""Command-line entry point for multiplex DMGI experiments."""

import argparse
import random

import numpy as np
import torch

from scheduler_utils import (
    DEFAULT_COSINE_MIN_LR_RATIO,
    DEFAULT_COSINE_T0,
    DEFAULT_COSINE_T_MULT,
    DEFAULT_LR_SCHEDULER,
    resolve_scheduler_defaults,
    validate_scheduler_args,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train a PyG-based multiplex DMGI model")
    parser.add_argument("--embedder", default="DMGI", choices=("DMGI",))
    parser.add_argument("--dataset", default="ADNI")
    parser.add_argument("--metapaths", default="type0,type1,type2,type3")
    parser.add_argument("--nb_epochs", type=int, default=10000)
    parser.add_argument("--hid_units", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--l2_coef", type=float, default=0.0001)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--reg_coef", type=float, default=0.001)
    parser.add_argument("--sup_coef", type=float, default=0.1)
    parser.add_argument("--sc", type=float, default=3.0)
    parser.add_argument("--gpu_num", default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--nheads", type=int, default=1)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--isSemi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--isBias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--isAttn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint_name")
    parser.add_argument("--lr_scheduler", choices=("none", "cosine_warm_restarts"), default=DEFAULT_LR_SCHEDULER)
    parser.add_argument("--cosine_t0", type=int, default=DEFAULT_COSINE_T0)
    parser.add_argument("--cosine_t_mult", type=int, default=DEFAULT_COSINE_T_MULT)
    parser.add_argument("--cosine_eta_min", type=float)
    return parser


def configure_reproducibility(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    configure_reproducibility()
    args, _unknown = build_parser().parse_known_args()
    resolve_scheduler_defaults(args)
    validate_scheduler_args(args)
    from models.DMGI import DMGI
    DMGI(args).training()


if __name__ == "__main__":
    main()
