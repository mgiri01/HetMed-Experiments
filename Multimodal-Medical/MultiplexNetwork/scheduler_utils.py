import torch


DEFAULT_LR_SCHEDULER = "cosine_warm_restarts"
DEFAULT_COSINE_T0 = 400
DEFAULT_COSINE_T_MULT = 2
DEFAULT_COSINE_MIN_LR_RATIO = 0.1


def resolve_scheduler_defaults(args):
    cosine_eta_min = getattr(args, "cosine_eta_min", None)
    if cosine_eta_min is None:
        args.cosine_eta_min = float(args.lr) * DEFAULT_COSINE_MIN_LR_RATIO
    else:
        args.cosine_eta_min = float(cosine_eta_min)


def validate_scheduler_args(args):
    scheduler_name = getattr(args, "lr_scheduler", "none")
    valid_schedulers = {"none", "cosine_warm_restarts"}
    if scheduler_name not in valid_schedulers:
        raise ValueError(
            "Unsupported lr_scheduler '{}'. Available options: {}.".format(
                scheduler_name, ", ".join(sorted(valid_schedulers))
            )
        )

    if scheduler_name == "none":
        return

    if int(args.cosine_t0) < 1:
        raise ValueError("--cosine_t0 must be at least 1 epoch.")
    if int(args.cosine_t_mult) < 1:
        raise ValueError("--cosine_t_mult must be at least 1.")
    if float(args.cosine_eta_min) < 0.0:
        raise ValueError("--cosine_eta_min must be non-negative.")
    if float(args.cosine_eta_min) >= float(args.lr):
        raise ValueError(
            "--cosine_eta_min must stay below --lr so cosine annealing can decay."
        )


def build_lr_scheduler(args, optimiser):
    if getattr(args, "lr_scheduler", "none") == "none":
        return None

    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser,
        T_0=int(args.cosine_t0),
        T_mult=int(args.cosine_t_mult),
        eta_min=float(args.cosine_eta_min),
    )


def scheduler_config_to_dict(args):
    scheduler_name = getattr(args, "lr_scheduler", "none")
    config = {"name": scheduler_name}
    if scheduler_name == "none":
        return config

    lr = float(args.lr)
    eta_min = float(args.cosine_eta_min)
    config.update(
        {
            "T_0": int(args.cosine_t0),
            "T_mult": int(args.cosine_t_mult),
            "eta_min": eta_min,
            "eta_min_ratio_to_lr": float(eta_min / lr) if lr > 0.0 else None,
        }
    )
    return config
