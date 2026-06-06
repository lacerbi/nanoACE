from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ace import ACE, ACEConfig, Batch, Tokens, Variable, cat_tokens, QUERY, VALUE


MU_RANGE = (-1.5, 1.5)
LOGSIG_RANGE = (math.log(0.15), math.log(1.25))
EVAL_Y = (0.6780859231948853, 0.852228581905365, 2.016355037689209)
EVAL_TRUE_MU = 0.891850471496582
EVAL_TRUE_LOGSIG = -0.4232509136199951


@dataclass
class ToyBatch:
    """A sampled Gaussian-toy ACE batch plus fields used by the oracle."""

    batch: Batch
    y_context: torch.Tensor
    mu: torch.Tensor
    log_sigma: torch.Tensor


@dataclass
class Diagnostic:
    """Held-out posterior comparison for one toy problem."""

    toy: ToyBatch
    oracle: dict[str, torch.Tensor]
    mu_logp: torch.Tensor
    logsig_logp: torch.Tensor
    joint_logp: torch.Tensor
    y_grid: torch.Tensor
    oracle_y_pred: torch.Tensor
    model_y_logp: torch.Tensor
    metrics: dict[str, float]


def variables(n_bins: int) -> list[Variable]:
    """Schema for the toy task.

    `y` is observed data. `mu` and `log_sigma` are interpretable continuous
    latents sampled from fixed priors. We model sigma in log-space because it is
    positive and the Gaussian likelihood is better behaved there.
    """

    return [
        Variable("y", "data", "continuous"),
        Variable("mu", "latent", "continuous", prior_range=MU_RANGE, prior_bins=n_bins),
        Variable("log_sigma", "latent", "continuous", transform="log", prior_range=LOGSIG_RANGE, prior_bins=n_bins),
    ]


def fixed_prior(
    batch: int,
    bins: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Uniform latent prior used by the Gaussian toy oracle and sampler."""

    return torch.full((batch, bins), 1.0 / bins, device=device)


def make_tokens(
    *,
    var_id: torch.Tensor,
    value: torch.Tensor,
    prior: torch.Tensor,
    mode: torch.Tensor,
    mask: torch.Tensor,
    x_dim: int,
) -> Tokens:
    """Construct a `Tokens` object for this scalar toy.

    The toy has no data covariates, so `x` is always zero. GP-1D will be the
    first example where data tokens use nontrivial `x`.
    """

    b, t = var_id.shape
    return Tokens(
        var_id=var_id.long(),
        x=torch.zeros(b, t, x_dim, device=value.device, dtype=value.dtype),
        value=value,
        value_index=torch.zeros(b, t, device=value.device, dtype=torch.long),
        prior=prior,
        mode=mode.long(),
        mask=mask.bool(),
    )


def sample_toy_batch(
    vars_: list[Variable],
    *,
    batch_size: int,
    max_context: int,
    min_context: int,
    data_targets: int,
    bins: int,
    device: torch.device | str,
    latent_context_prob: float = 0.0,
) -> ToyBatch:
    """Generate one online training/eval batch.

    Context contains a variable number of observed `y` samples. Targets ask for
    the two latents and a few held-out `y` values, which keeps the model trained
    on both data prediction and latent prediction. During training, one latent
    can also be revealed as a context VALUE token so autoregressive latent
    conditioning is in-distribution.
    """

    mu = torch.empty(batch_size, device=device).uniform_(*MU_RANGE)
    log_sigma = torch.empty(batch_size, device=device).uniform_(*LOGSIG_RANGE)
    sigma = log_sigma.exp()

    total_y = max_context + data_targets
    y = mu[:, None] + sigma[:, None] * torch.randn(batch_size, total_y, device=device)
    n_ctx = torch.randint(min_context, max_context + 1, (batch_size,), device=device)
    ar = torch.arange(max_context, device=device)[None, :]
    if latent_context_prob > 0.0:
        reveal = torch.rand(batch_size, device=device) < latent_context_prob
        reveal_mu = reveal & (torch.rand(batch_size, device=device) < 0.5)
        reveal_logsig = reveal & ~reveal_mu
    else:
        reveal_mu = torch.zeros(batch_size, device=device, dtype=torch.bool)
        reveal_logsig = torch.zeros(batch_size, device=device, dtype=torch.bool)

    ctx_t = max_context + 2
    mu_value_pos = max_context
    logsig_value_pos = max_context + 1
    ctx_var = torch.zeros(batch_size, ctx_t, device=device, dtype=torch.long)
    ctx_var[:, mu_value_pos] = 1
    ctx_var[:, logsig_value_pos] = 2
    ctx_value = torch.zeros(batch_size, ctx_t, device=device)
    ctx_value[:, :max_context] = y[:, :max_context]
    ctx_value[:, mu_value_pos] = mu
    ctx_value[:, logsig_value_pos] = log_sigma
    ctx_prior = torch.zeros(batch_size, ctx_t, bins, device=device)
    ctx_mode = torch.full((batch_size, ctx_t), VALUE, device=device)
    ctx_mask = torch.zeros(batch_size, ctx_t, device=device, dtype=torch.bool)
    ctx_mask[:, :max_context] = ar < n_ctx[:, None]
    ctx_mask[:, mu_value_pos] = reveal_mu
    ctx_mask[:, logsig_value_pos] = reveal_logsig
    context = make_tokens(
        var_id=ctx_var,
        value=ctx_value,
        prior=ctx_prior,
        mode=ctx_mode,
        mask=ctx_mask,
        x_dim=1,
    )

    tgt_t = 2 + data_targets
    tgt_var = torch.zeros(batch_size, tgt_t, device=device, dtype=torch.long)
    tgt_var[:, 0] = 1
    tgt_var[:, 1] = 2
    tgt_value = torch.zeros(batch_size, tgt_t, device=device)
    tgt_value[:, 0] = mu
    tgt_value[:, 1] = log_sigma
    tgt_value[:, 2:] = y[:, max_context:]
    tgt_mask = torch.ones(batch_size, tgt_t, device=device, dtype=torch.bool)
    tgt_mask[:, 0] = ~reveal_mu
    tgt_mask[:, 1] = ~reveal_logsig
    target = make_tokens(
        var_id=tgt_var,
        value=tgt_value,
        prior=torch.zeros(batch_size, tgt_t, bins, device=device),
        mode=torch.full((batch_size, tgt_t), QUERY, device=device),
        mask=tgt_mask,
        x_dim=1,
    )
    return ToyBatch(Batch(vars_, context, target), y[:, :max_context], mu, log_sigma)


def _repeat_tokens(tokens: Tokens, repeats: int) -> Tokens:
    """Repeat one context batch so a whole grid can be queried in parallel."""

    return Tokens(
        var_id=tokens.var_id.repeat(repeats, 1),
        x=tokens.x.repeat(repeats, 1, 1),
        value=tokens.value.repeat(repeats, 1),
        value_index=tokens.value_index.repeat(repeats, 1),
        prior=tokens.prior.repeat(repeats, 1, 1),
        mode=tokens.mode.repeat(repeats, 1),
        mask=tokens.mask.repeat(repeats, 1),
    )


def value_token(
    model: ACE,
    *,
    var_id: int,
    values: torch.Tensor,
) -> Tokens:
    """Build VALUE tokens for a latent grid value.

    This is used by the autoregressive diagnostic: after predicting one latent,
    we append a concrete sampled/grid value to the context before querying the
    next latent.
    """

    b = values.numel()
    return make_tokens(
        var_id=torch.full((b, 1), var_id, device=values.device),
        value=values[:, None],
        prior=torch.zeros(b, 1, model.cfg.prior_bins, device=values.device),
        mode=torch.full((b, 1), VALUE, device=values.device),
        mask=torch.ones(b, 1, device=values.device, dtype=torch.bool),
        x_dim=model.cfg.x_dim,
    )


def query_log_density(
    model: ACE,
    batch: Batch,
    var_id: int,
    values: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ACE's 1D marginal log density for one latent over a grid."""

    b = values.numel()
    prior = torch.zeros(b, 1, model.cfg.prior_bins, device=values.device)
    target = make_tokens(
        var_id=torch.full((b, 1), var_id, device=values.device),
        value=values[:, None],
        prior=prior,
        mode=torch.full((b, 1), QUERY, device=values.device),
        mask=torch.ones(b, 1, device=values.device, dtype=torch.bool),
        x_dim=model.cfg.x_dim,
    )
    rep = Batch(batch.variables, _repeat_tokens(batch.context, b), target)
    return model(rep).log_prob(target).squeeze(1)


def conditional_log_density(
    model: ACE,
    batch: Batch,
    *,
    known_var: int,
    known_values: torch.Tensor,
    query_var: int,
    query_values: torch.Tensor,
) -> torch.Tensor:
    """Grid of log p(query_var | context, known_var=value).

    Returns `[len(known_values), len(query_values)]`. We loop over known values
    to keep the intermediate token batch small enough to remain readable.
    """

    rows = []
    q = query_values.numel()
    for known in known_values:
        known_tok = value_token(model, var_id=known_var, values=known.expand(q))
        context = cat_tokens([_repeat_tokens(batch.context, q), known_tok])
        target = make_tokens(
            var_id=torch.full((q, 1), query_var, device=query_values.device),
            value=query_values[:, None],
            prior=torch.zeros(q, 1, model.cfg.prior_bins, device=query_values.device),
            mode=torch.full((q, 1), QUERY, device=query_values.device),
            mask=torch.ones(q, 1, device=query_values.device, dtype=torch.bool),
            x_dim=model.cfg.x_dim,
        )
        rows.append(model(Batch(batch.variables, context, target)).log_prob(target).squeeze(1))
    return torch.stack(rows, dim=0)


def ar_joint_log_density(
    model: ACE,
    batch: Batch,
    mu_grid: torch.Tensor,
    logsig_grid: torch.Tensor,
) -> torch.Tensor:
    """Autoregressive approximation to p(mu, log_sigma | context).

    We average two factorizations in probability space:

    - p(mu | D) p(log_sigma | D, mu)
    - p(log_sigma | D) p(mu | D, log_sigma)

    This is a small Janossy-style symmetrization that avoids making the plot
    depend entirely on one arbitrary latent order.
    """

    log_mu = query_log_density(model, batch, 1, mu_grid)
    log_s = query_log_density(model, batch, 2, logsig_grid)
    log_s_given_mu = conditional_log_density(
        model,
        batch,
        known_var=1,
        known_values=mu_grid,
        query_var=2,
        query_values=logsig_grid,
    )
    log_mu_given_s = conditional_log_density(
        model,
        batch,
        known_var=2,
        known_values=logsig_grid,
        query_var=1,
        query_values=mu_grid,
    ).transpose(0, 1)
    joint_1 = log_mu[:, None] + log_s_given_mu
    joint_2 = log_s[None, :] + log_mu_given_s
    joint = torch.logsumexp(torch.stack([joint_1, joint_2], dim=0), dim=0) - math.log(2.0)
    return joint - torch.logsumexp(joint.reshape(-1), dim=0)


def analytic_posterior(
    y_obs: torch.Tensor,
    *,
    bins: int,
) -> dict[str, torch.Tensor]:
    """Exact grid posterior for the Gaussian toy.

    The prior is factorized over (`mu`, `log_sigma`) and represented on the same
    grids used by the toy sampler. Given observed data, Bayes' rule gives a
    normalized 2D grid posterior. We return the two 1D marginals and moments for
    a compact diagnostic.
    """

    device = y_obs.device
    mu_grid = torch.linspace(MU_RANGE[0], MU_RANGE[1], bins, device=device)
    logsig_grid = torch.linspace(LOGSIG_RANGE[0], LOGSIG_RANGE[1], bins, device=device)
    prior_mu = fixed_prior(1, bins, device=device)[0]
    prior_logsig = fixed_prior(1, bins, device=device)[0]
    sigma_grid = logsig_grid.exp()
    y = y_obs[None, None, :]
    mu = mu_grid[:, None, None]
    sigma = sigma_grid[None, :, None]
    loglike = (-0.5 * ((y - mu) / sigma).pow(2) - sigma.log() - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
    logprior = prior_mu.log().clamp_min(-1e30)[:, None] + prior_logsig.log().clamp_min(-1e30)[None, :]
    logpost = logprior + loglike
    post = (logpost - torch.logsumexp(logpost.reshape(-1), dim=0)).exp()
    pmu = post.sum(dim=1)
    ps = post.sum(dim=0)
    mu_mean = (pmu * mu_grid).sum()
    logsig_mean = (ps * logsig_grid).sum()
    mu_std = (pmu * (mu_grid - mu_mean).pow(2)).sum().sqrt()
    logsig_std = (ps * (logsig_grid - logsig_mean).pow(2)).sum().sqrt()
    cov = (post * (mu_grid[:, None] - mu_mean) * (logsig_grid[None, :] - logsig_mean)).sum()
    corr = cov / (mu_std * logsig_std).clamp_min(1e-8)
    return {
        "mu_grid": mu_grid,
        "logsig_grid": logsig_grid,
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "logsig_mean": logsig_mean,
        "logsig_std": logsig_std,
        "corr": corr,
        "pmu": pmu,
        "plogsig": ps,
        "post": post,
    }


def predictive_grid(
    oracle: dict[str, torch.Tensor],
    y_obs: torch.Tensor,
    *,
    points: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic posterior predictive density for a new Gaussian observation.

    This is not a Gaussian density. It marginalizes over the full posterior grid:
    `sum_{mu, sigma} p(mu, log_sigma | D) Normal(y_new | mu, sigma)`.
    """

    mu_grid = oracle["mu_grid"]
    logsig_grid = oracle["logsig_grid"]
    sigma_grid = logsig_grid.exp()
    post = oracle["post"]
    mu = mu_grid[:, None]
    sigma = sigma_grid[None, :]

    pred_mean = (post * mu).sum()
    pred_second = (post * (sigma.pow(2) + mu.pow(2))).sum()
    pred_std = (pred_second - pred_mean.pow(2)).clamp_min(1e-8).sqrt()
    lo = torch.minimum(y_obs.min(), pred_mean - 5.0 * pred_std)
    hi = torch.maximum(y_obs.max(), pred_mean + 5.0 * pred_std)
    y_grid = torch.linspace(float(lo), float(hi), points, device=mu_grid.device)

    y = y_grid[None, None, :]
    comp = -0.5 * ((y - mu[:, :, None]) / sigma[:, :, None]).pow(2)
    comp = comp.exp() / (sigma[:, :, None] * math.sqrt(2.0 * math.pi))
    pred = (post[:, :, None] * comp).sum(dim=(0, 1))
    return y_grid, pred


def normalized_moments(grid: torch.Tensor, log_density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std of a 1D density evaluated on an evenly spaced grid."""

    p = (log_density - torch.logsumexp(log_density, dim=0)).exp()
    mean = (p * grid).sum()
    std = (p * (grid - mean).pow(2)).sum().sqrt()
    return mean, std


def build_model(args: argparse.Namespace, device: torch.device) -> ACE:
    """Construct the toy ACE model from CLI hyperparameters."""

    vars_ = variables(args.bins)
    cfg = ACEConfig(
        x_dim=1,
        prior_bins=args.bins,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        mlp_hidden=args.hidden,
        head_hidden=args.hidden,
        mdn_components=args.components,
    )
    return ACE(vars_, cfg).to(device)


def save_checkpoint(model: ACE, path: str | Path, args: argparse.Namespace) -> None:
    """Save a lightweight demo checkpoint.

    The checkpoint is a convenience artifact only: it stores the model config,
    seed, and `state_dict`, and can always be regenerated by retraining.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(model.cfg), "seed": args.seed, "state_dict": model.state_dict()}, path)
    print(f"saved checkpoint: {path}")


def load_checkpoint(path: str | Path, device: torch.device) -> ACE:
    """Load a demo checkpoint saved by `save_checkpoint`."""

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ACEConfig(**payload["cfg"])
    model = ACE(variables(cfg.prior_bins), cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


def train(args: argparse.Namespace, model: ACE | None = None) -> ACE:
    """Train ACE online on freshly sampled Gaussian-toy batches."""

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = build_model(args, device) if model is None else model
    vars_ = model.variables
    bins = model.cfg.prior_bins
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        toy = sample_toy_batch(
            vars_,
            batch_size=args.batch_size,
            max_context=args.max_context,
            min_context=args.min_context,
            data_targets=args.data_targets,
            bins=bins,
            device=device,
            latent_context_prob=args.latent_context_prob,
        )
        loss = model.loss(toy.batch, latent_weight=args.latent_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}")
    return model


def fixed_eval_batch(vars_: list[Variable], *, bins: int, device: torch.device | str) -> ToyBatch:
    """The fixed three-observation problem used by the diagnostic plot."""

    y_obs = torch.tensor(EVAL_Y, device=device)
    mu = torch.tensor([EVAL_TRUE_MU], device=device)
    log_sigma = torch.tensor([EVAL_TRUE_LOGSIG], device=device)
    n = y_obs.numel()

    context = make_tokens(
        var_id=torch.zeros(1, n, device=device, dtype=torch.long),
        value=y_obs[None, :],
        prior=torch.zeros(1, n, bins, device=device),
        mode=torch.full((1, n), VALUE, device=device),
        mask=torch.ones(1, n, device=device, dtype=torch.bool),
        x_dim=1,
    )
    target = make_tokens(
        var_id=torch.tensor([[1, 2]], device=device),
        value=torch.stack([mu, log_sigma], dim=1),
        prior=torch.zeros(1, 2, bins, device=device),
        mode=torch.full((1, 2), QUERY, device=device),
        mask=torch.ones(1, 2, device=device, dtype=torch.bool),
        x_dim=1,
    )
    return ToyBatch(Batch(vars_, context, target), y_obs[None, :], mu, log_sigma)


@torch.no_grad()
def evaluate(model: ACE, args: argparse.Namespace) -> Diagnostic:
    """Compare ACE's posterior marginals and AR joint with the analytic oracle."""

    device = next(model.parameters()).device
    vars_ = model.variables
    bins = model.cfg.prior_bins
    toy = fixed_eval_batch(vars_, bins=bins, device=device)
    true = analytic_posterior(toy.y_context[0], bins=bins)
    eval_context = int(toy.y_context.shape[1])
    mu_grid = true["mu_grid"]
    logsig_grid = true["logsig_grid"]
    mu_logp = query_log_density(model, toy.batch, 1, mu_grid)
    logsig_logp = query_log_density(model, toy.batch, 2, logsig_grid)
    joint_logp = ar_joint_log_density(model, toy.batch, mu_grid, logsig_grid)
    y_grid, oracle_y_pred = predictive_grid(true, toy.y_context[0])
    model_y_logp = query_log_density(model, toy.batch, 0, y_grid)
    mu_mean, mu_std = normalized_moments(mu_grid, mu_logp)
    logsig_mean, logsig_std = normalized_moments(logsig_grid, logsig_logp)
    oracle_y_mean, oracle_y_std = normalized_moments(y_grid, oracle_y_pred.clamp_min(1e-30).log())
    model_y_mean, model_y_std = normalized_moments(y_grid, model_y_logp)
    p_joint = joint_logp.exp()
    joint_mu_mean = (p_joint.sum(dim=1) * mu_grid).sum()
    joint_s_mean = (p_joint.sum(dim=0) * logsig_grid).sum()
    metrics = {
        "mu_mean_abs_err": float((mu_mean - true["mu_mean"]).abs()),
        "mu_std_abs_err": float((mu_std - true["mu_std"]).abs()),
        "logsig_mean_abs_err": float((logsig_mean - true["logsig_mean"]).abs()),
        "logsig_std_abs_err": float((logsig_std - true["logsig_std"]).abs()),
        "joint_mu_mean_abs_err": float((joint_mu_mean - true["mu_mean"]).abs()),
        "joint_logsig_mean_abs_err": float((joint_s_mean - true["logsig_mean"]).abs()),
        "pred_y_mean_abs_err": float((model_y_mean - oracle_y_mean).abs()),
        "pred_y_std_abs_err": float((model_y_std - oracle_y_std).abs()),
        "oracle_corr": float(true["corr"]),
    }
    print("\nHeld-out Gaussian toy posterior moments")
    print(f"eval context    fixed {eval_context}-observation case")
    print(f"truth mu        {float(toy.mu[0]): .3f}")
    print(f"truth sigma     {float(toy.log_sigma[0].exp()): .3f}")
    print(f"oracle corr     {float(true['corr']): .3f}")
    print(f"oracle mu       mean {float(true['mu_mean']): .3f}  std {float(true['mu_std']): .3f}")
    print(f"model  mu       mean {float(mu_mean): .3f}  std {float(mu_std): .3f}")
    print(f"oracle log_sig  mean {float(true['logsig_mean']): .3f}  std {float(true['logsig_std']): .3f}")
    print(f"model  log_sig  mean {float(logsig_mean): .3f}  std {float(logsig_std): .3f}")
    print(f"oracle pred y   mean {float(oracle_y_mean): .3f}  std {float(oracle_y_std): .3f}")
    print(f"model  pred y   mean {float(model_y_mean): .3f}  std {float(model_y_std): .3f}")
    print(f"AR joint mean   mu {float(joint_mu_mean): .3f}  log_sig {float(joint_s_mean): .3f}")
    return Diagnostic(toy, true, mu_logp, logsig_logp, joint_logp, y_grid, oracle_y_pred, model_y_logp, metrics)


def plot_diagnostic(diag: Diagnostic, path: str | Path) -> None:
    """Save a compact posterior diagnostic figure."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mu = diag.oracle["mu_grid"].detach().cpu()
    logsig = diag.oracle["logsig_grid"].detach().cpu()
    oracle_joint = diag.oracle["post"].detach().cpu().T
    model_joint = diag.joint_logp.exp().detach().cpu().T
    oracle_mu = diag.oracle["pmu"].detach().cpu()
    oracle_s = diag.oracle["plogsig"].detach().cpu()
    model_mu = (diag.mu_logp - torch.logsumexp(diag.mu_logp, dim=0)).exp().detach().cpu()
    model_s = (diag.logsig_logp - torch.logsumexp(diag.logsig_logp, dim=0)).exp().detach().cpu()
    y_grid = diag.y_grid.detach().cpu()
    oracle_y = diag.oracle_y_pred.detach().cpu()
    model_y = diag.model_y_logp.exp().detach().cpu()
    y_obs = diag.toy.y_context[0].detach().cpu()

    fig = plt.figure(figsize=(9, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85])
    ax_oracle_joint = fig.add_subplot(gs[0, 0])
    ax_model_joint = fig.add_subplot(gs[0, 1])
    ax_mu = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])
    ax_pred = fig.add_subplot(gs[2, :])

    extent = [float(mu[0]), float(mu[-1]), float(logsig[0]), float(logsig[-1])]
    ax_oracle_joint.imshow(oracle_joint, origin="lower", aspect="auto", extent=extent)
    ax_oracle_joint.set_title("Oracle joint")
    ax_model_joint.imshow(model_joint, origin="lower", aspect="auto", extent=extent)
    ax_model_joint.set_title("ACE AR joint")
    for ax in (ax_oracle_joint, ax_model_joint):
        ax.set_xlabel("mu")
        ax.set_ylabel("log_sigma")

    ax_mu.plot(mu, oracle_mu, label="oracle")
    ax_mu.plot(mu, model_mu, label="ACE")
    ax_mu.set_title("mu marginal")
    ax_mu.set_xlabel("mu")
    ax_mu.legend()

    ax_s.plot(logsig, oracle_s, label="oracle")
    ax_s.plot(logsig, model_s, label="ACE")
    ax_s.set_title("log_sigma marginal")
    ax_s.set_xlabel("log_sigma")
    ax_s.legend()

    ax_pred.plot(y_grid, oracle_y, label="oracle")
    ax_pred.plot(y_grid, model_y, label="ACE")
    for y in y_obs:
        ax_pred.axvline(float(y), color="0.25", alpha=0.2, linewidth=1.0)
    ax_pred.set_title("posterior predictive")
    ax_pred.set_xlabel("new y")
    ax_pred.set_ylabel("density")
    ax_pred.legend()

    y_tokens = (diag.toy.batch.context.var_id == 0) & diag.toy.batch.context.mask
    eval_n = int(y_tokens[0].sum().item())
    fig.suptitle(f"eval N={eval_n}, oracle corr={float(diag.oracle['corr']):.2f}")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved diagnostic plot: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=64)
    p.add_argument("--max-context", type=int, default=16)
    p.add_argument("--min-context", type=int, default=2)
    p.add_argument("--data-targets", type=int, default=4)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent-weight", type=float, default=2.0)
    p.add_argument("--latent-context-prob", type=float, default=0.25)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--plot-path", default="artifacts/gaussian_toy.png")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--save-checkpoint", default="")
    p.add_argument("--load-checkpoint", default="")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.load_checkpoint:
        model = load_checkpoint(args.load_checkpoint, device)
    elif args.eval_only:
        raise SystemExit("--eval-only requires --load-checkpoint")
    else:
        model = None

    if not args.eval_only:
        model = train(args, model)
    assert model is not None
    diag = evaluate(model, args)
    if args.save_checkpoint:
        save_checkpoint(model, args.save_checkpoint, args)
    if not args.no_plot and args.plot_path:
        plot_diagnostic(diag, args.plot_path)


if __name__ == "__main__":
    main()
