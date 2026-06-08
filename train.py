"""Shared training utilities for ACE models.

Callers own their variable schema, batch source, diagnostics, plots, and entry point.
This module provides the common pieces:

- `common_parser()` and `apply_config_file()`: shared CLI arguments and optional YAML
  defaults, with explicit CLI flags taking precedence.
- `build_model()`: construct an `ACE` from CLI hyperparameters and `variables()`.
- `TrainConfig`: typed training settings for `fit`.
- `fit()`: one optimization loop for any `sample_batch(step) -> Batch` source. Online
  samplers and `data.PoolReader` use the same interface.
- `save_checkpoint()`, `load_checkpoint()`, and `load_train_state()`: model and resume
  checkpoint helpers.

`fit` reseeds torch with `mix_seed(seed, step)` before each batch, so the data stream is
a pure function of `(seed, step)` and resumed runs replay the same batches as uninterrupted
runs. The caller seeds model construction before calling `build_model`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from ace import ACE, ACEConfig, Batch, Variable, mix_seed


# --------------------------------------------------------------------------- #
# CLI: shared argument parent + light YAML config
# --------------------------------------------------------------------------- #


def common_parser() -> argparse.ArgumentParser:
    """Return an argparse parent for ACE training CLIs.

    Holds common run-control, model, loss, checkpoint, schedule/resume, and
    `--config` flags. Defaults here are neutral; a caller can override them with
    `parser.set_defaults(...)` and then add task-specific arguments. Use as::

        p = argparse.ArgumentParser(parents=[common_parser()], description=...)
        p.set_defaults(d_model=192, batch_size=64, ...)   # caller defaults
        p.add_argument("--jitter", ...)                   # task-specific
        args = apply_config_file(p)                        # layers --config YAML
    """

    p = argparse.ArgumentParser(add_help=False)

    # Run control - shared defaults for ACE training scripts.
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent-weight", type=float, default=2.0)
    p.add_argument("--data-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--latent-context-prob",
        type=float,
        default=0.5,
        help="P(reveal any latents) per task; the revealed subset uses the shared mixture DGP",
    )
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--save-checkpoint", default="")
    p.add_argument("--load-checkpoint", default="")
    p.add_argument("--eval-only", action="store_true")

    # Caller-overridden - neutral defaults; task CLIs can call set_defaults().
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-context", type=int, default=12)
    p.add_argument("--min-context", type=int, default=4)
    p.add_argument("--data-targets", type=int, default=16)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--plot-path", default="artifacts/diagnostic.png")

    # Training schedule + simple resume.
    p.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    p.add_argument("--warmup", type=int, default=0, help="linear warmup steps before cosine decay")
    p.add_argument("--resume", default="", help="resume optimizer/scheduler/step from this checkpoint")
    p.add_argument(
        "--ckpt-every",
        type=int,
        default=0,
        help="write a resumable checkpoint to --save-checkpoint every N steps (0 = off)",
    )

    # Config file (light YAML; overridden by explicit CLI flags).
    p.add_argument("--config", default="", help="YAML file of defaults; explicit CLI flags win")
    return p


def _load_yaml(path: str | Path) -> dict:
    import yaml  # lazy: only needed when --config is passed

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config file {path} must be a YAML mapping, got {type(data).__name__}")
    return data


def apply_config_file(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse `parser` with an optional `--config` YAML layered under CLI flags.

    Precedence: parser defaults (including caller `set_defaults`) < `--config`
    YAML < explicit CLI flags. YAML keys must be argparse *dest* names
    (underscored, e.g. `latent_context_prob: 0.5`); unknown keys are rejected so
    typos fail loudly instead of being silently ignored.
    """

    pre, _ = parser.parse_known_args()
    cfg_path = getattr(pre, "config", "")
    if cfg_path:
        data = _load_yaml(cfg_path)
        actions = {a.dest: a for a in parser._actions if a.dest != "help"}
        unknown = sorted(set(data) - set(actions))
        if unknown:
            raise SystemExit(f"unknown key(s) in {cfg_path}: {unknown}")
        # `set_defaults` bypasses argparse's `type=`/`choices`, so coerce + validate
        # here. Without this, e.g. YAML `lr: 3e-4` arrives as the string "3e-4"
        # (YAML 1.1 only treats `3.0e-4` as a float) and reaches the optimizer unconverted.
        for key, value in data.items():
            action = actions[key]
            if action.type is not None and value is not None and not isinstance(value, bool):
                try:
                    value = action.type(value)
                except (ValueError, TypeError) as exc:
                    raise SystemExit(f"config {cfg_path}: bad value for {key!r}: {value!r} ({exc})")
            if action.choices is not None and value not in action.choices:
                raise SystemExit(f"config {cfg_path}: {key!r}={value!r} not in {list(action.choices)}")
            data[key] = value
        parser.set_defaults(**data)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Model construction + training config
# --------------------------------------------------------------------------- #


def build_model(args, variables: Sequence[Variable], device: torch.device | str) -> ACE:
    """Construct an ACE model from CLI hyperparameters and a variable schema."""

    cfg = ACEConfig(
        x_dim=getattr(args, "x_dim", 1),
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        mlp_hidden=args.hidden,
        head_hidden=args.hidden,
        mdn_components=args.components,
    )
    return ACE(list(variables), cfg).to(device)


@dataclass
class TrainConfig:
    """Typed training knobs for `fit`, decoupled from argparse."""

    steps: int
    lr: float
    latent_weight: float = 1.0
    data_weight: float = 1.0
    grad_clip: float = 1.0
    log_every: int = 100
    lr_schedule: str = "cosine"
    warmup: int = 0

    @classmethod
    def from_args(cls, args) -> "TrainConfig":
        return cls(
            steps=args.steps,
            lr=args.lr,
            latent_weight=args.latent_weight,
            data_weight=getattr(args, "data_weight", 1.0),
            grad_clip=getattr(args, "grad_clip", 1.0),
            log_every=args.log_every,
            lr_schedule=getattr(args, "lr_schedule", "cosine"),
            warmup=getattr(args, "warmup", 0),
        )


def _build_scheduler(opt: torch.optim.Optimizer, cfg: TrainConfig):
    """Cosine LR with optional linear warmup, or None for a constant LR.

    Returns a `LambdaLR` whose multiplier ramps `0 -> 1` over `warmup` steps, then
    follows a cosine from `1 -> 0` over the remaining steps to `cfg.steps`. Cosine
    `T_max` is the run's total budget, so a resumed run must keep the same
    `--steps` for the curve to line up with the restored scheduler state.
    """

    if cfg.lr_schedule == "constant":
        return None
    warmup = max(0, cfg.warmup)
    total = max(1, cfg.steps)

    def lr_lambda(completed: int) -> float:
        step = completed + 1  # 1-based current step
        if warmup and step <= warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def save_checkpoint(
    path: str | Path,
    model: ACE,
    *,
    seed: int,
    config: dict | None = None,
    opt: torch.optim.Optimizer | None = None,
    sched=None,
    step: int | None = None,
) -> None:
    """Save a checkpoint.

    Always writes `{cfg, seed, state_dict}`. Adds `config` when given, and
    `{optimizer, scheduler, step}` when those are passed for resume. Extra keys are
    additive: `load_checkpoint` reads only the model keys.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"cfg": asdict(model.cfg), "seed": seed, "state_dict": model.state_dict()}
    if config is not None:
        payload["config"] = dict(config)
    if opt is not None:
        payload["optimizer"] = opt.state_dict()
    if sched is not None:
        payload["scheduler"] = sched.state_dict()
    if step is not None:
        payload["step"] = int(step)
    torch.save(payload, path)
    print(f"saved checkpoint: {path}")


def load_checkpoint(path: str | Path, device: torch.device | str, variables: Sequence[Variable]) -> ACE:
    """Load a model from a checkpoint, given its variable schema.

    Reads only `cfg` and `state_dict`; resume/provenance keys are ignored here.
    """

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ACEConfig(**payload["cfg"])
    model = ACE(list(variables), cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def load_train_state(payload: dict, opt: torch.optim.Optimizer, sched=None) -> int:
    """Restore optimizer/scheduler state from a checkpoint payload; return the next step.

    Tolerates model-only / legacy checkpoints (missing resume keys), in which case
    nothing is restored and the next step is 1.
    """

    if "optimizer" in payload:
        opt.load_state_dict(payload["optimizer"])
    if sched is not None and payload.get("scheduler") is not None:
        sched.load_state_dict(payload["scheduler"])
    return int(payload.get("step", 0)) + 1


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def fit(
    model: ACE,
    sample_batch: Callable[[int], Batch],
    cfg: TrainConfig,
    *,
    resume_state: dict | None = None,
    seed: int = 0,
    checkpoint_path: str | Path | None = None,
    ckpt_every: int = 0,
) -> ACE:
    """Train `model`; one `Batch` per step from `sample_batch(step)`.

    Uses Adam + grad clip, with constant or cosine LR per `cfg.lr_schedule`. At the top
    of every step, `fit` calls `torch.manual_seed(mix_seed(seed, step))`; the batch at a
    given step is therefore a pure function of `(seed, step)`, independent of model-init
    RNG use and exact across resume.

    If `resume_state` has optimizer/scheduler/step keys, training resumes from that step.
    Keep the same `--steps` when using cosine LR so the restored scheduler state matches
    the intended curve.

    The caller owns model-construction seeding. `fit`'s per-step reseed is for the data
    stream; ACE's forward pass has no stochastic ops.

    If `checkpoint_path` and `ckpt_every > 0` are set, resumable checkpoints are written
    during training.
    """

    if ckpt_every and not checkpoint_path:
        print("warning: --ckpt-every is set but no --save-checkpoint path; periodic checkpoints disabled")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = _build_scheduler(opt, cfg)

    start_step = 1
    if resume_state is not None:
        start_step = load_train_state(resume_state, opt, sched)
        if start_step > 1:  # genuinely resuming mid-run (resumable checkpoint had a step)
            prev_total = int(resume_state.get("config", {}).get("steps", cfg.steps))
            if prev_total != cfg.steps:
                print(
                    f"warning: resuming with --steps {cfg.steps} but the run was started with "
                    f"{prev_total}; the cosine LR curve will not line up. Use the original --steps."
                )

    for step in range(start_step, cfg.steps + 1):
        torch.manual_seed(mix_seed(seed, step))  # batch(step) = f(seed, step): reproducible + resume-exact
        batch = sample_batch(step)
        loss = model.loss(batch, data_weight=cfg.data_weight, latent_weight=cfg.latent_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
        if step == start_step or step % cfg.log_every == 0:
            print(f"step {step:5d}/{cfg.steps}  loss {loss.item():.4f}")
        if ckpt_every and checkpoint_path and step % ckpt_every == 0 and step < cfg.steps:
            # Record the TrainConfig as `config` so a later --resume can detect a
            # changed --steps (the cosine T_max) and warn; see the resume guard above.
            save_checkpoint(checkpoint_path, model, seed=seed, config=asdict(cfg), opt=opt, sched=sched, step=step)
    return model
