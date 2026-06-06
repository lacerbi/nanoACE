from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

from ace import ACE, ACEConfig, Batch, Tokens, Variable, PRIOR, QUERY, VALUE


MU_RANGE = (-1.5, 1.5)
LOGSIG_RANGE = (math.log(0.15), math.log(1.25))


@dataclass
class ToyBatch:
    """A sampled Gaussian-toy ACE batch plus fields used by the oracle."""

    batch: Batch
    y_context: torch.Tensor
    prior_mu: torch.Tensor
    prior_logsig: torch.Tensor
    mu: torch.Tensor
    log_sigma: torch.Tensor


def variables(n_bins: int) -> list[Variable]:
    """Schema for the toy task.

    `y` is observed data. `mu` and `log_sigma` are interpretable continuous
    latents with runtime priors. We model sigma in log-space because it is
    positive and the Gaussian likelihood is better behaved there.
    """

    return [
        Variable("y", "data", "continuous"),
        Variable("mu", "latent", "continuous", prior_range=MU_RANGE, prior_bins=n_bins),
        Variable("log_sigma", "latent", "continuous", transform="log", prior_range=LOGSIG_RANGE, prior_bins=n_bins),
    ]


def _normal_pdf(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    z = (x - loc) / scale.clamp_min(1e-6)
    return torch.exp(-0.5 * z.pow(2)) / (scale.clamp_min(1e-6) * math.sqrt(2.0 * math.pi))


def sample_smooth_priors(
    batch: int,
    bins: int,
    value_range: tuple[float, float],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample simple smooth histogram priors.

    This is intentionally much simpler than the paper's prior generator. It
    still trains the important behavior: the prior token can be sharp, broad, or
    mildly multimodal, and the model must combine it with observed data.
    """

    lo, hi = value_range
    centers = torch.linspace(lo, hi, bins, device=device)
    width = hi - lo
    k2 = torch.rand(batch, 1, device=device) < 0.35
    loc1 = lo + width * torch.rand(batch, 1, device=device)
    loc2 = lo + width * torch.rand(batch, 1, device=device)
    scale1 = width * (0.05 + 0.30 * torch.rand(batch, 1, device=device))
    scale2 = width * (0.05 + 0.30 * torch.rand(batch, 1, device=device))
    w = torch.rand(batch, 1, device=device)
    p1 = _normal_pdf(centers[None, :], loc1, scale1)
    p2 = _normal_pdf(centers[None, :], loc2, scale2)
    probs = torch.where(k2, w * p1 + (1.0 - w) * p2, p1)
    probs = probs + 0.03 / width
    return probs / probs.sum(dim=-1, keepdim=True)


def sample_from_hist(
    probs: torch.Tensor,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Draw a scalar from a normalized histogram over `value_range`."""

    lo, hi = value_range
    bins = probs.shape[-1]
    idx = torch.distributions.Categorical(probs=probs).sample()
    u = torch.rand_like(idx, dtype=probs.dtype)
    return lo + (idx.to(probs.dtype) + u) * ((hi - lo) / bins)


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
) -> ToyBatch:
    """Generate one online training/eval batch.

    Context contains a variable number of observed `y` samples plus two prior
    tokens, one for `mu` and one for `log_sigma`. Targets ask for the two
    latents and a few held-out `y` values, which keeps the model trained on both
    data prediction and latent prediction.
    """

    prior_mu = sample_smooth_priors(batch_size, bins, MU_RANGE, device=device)
    prior_logsig = sample_smooth_priors(batch_size, bins, LOGSIG_RANGE, device=device)
    mu = sample_from_hist(prior_mu, MU_RANGE)
    log_sigma = sample_from_hist(prior_logsig, LOGSIG_RANGE)
    sigma = log_sigma.exp()

    total_y = max_context + data_targets
    y = mu[:, None] + sigma[:, None] * torch.randn(batch_size, total_y, device=device)
    n_ctx = torch.randint(min_context, max_context + 1, (batch_size,), device=device)
    ar = torch.arange(max_context, device=device)[None, :]

    ctx_var = torch.zeros(batch_size, max_context + 2, device=device, dtype=torch.long)
    ctx_var[:, max_context] = 1
    ctx_var[:, max_context + 1] = 2
    ctx_value = torch.zeros(batch_size, max_context + 2, device=device)
    ctx_value[:, :max_context] = y[:, :max_context]
    ctx_prior = torch.zeros(batch_size, max_context + 2, bins, device=device)
    ctx_prior[:, max_context, :] = prior_mu
    ctx_prior[:, max_context + 1, :] = prior_logsig
    ctx_mode = torch.full((batch_size, max_context + 2), VALUE, device=device)
    ctx_mode[:, max_context:] = PRIOR
    ctx_mask = torch.zeros(batch_size, max_context + 2, device=device, dtype=torch.bool)
    ctx_mask[:, :max_context] = ar < n_ctx[:, None]
    ctx_mask[:, max_context:] = True
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
    target = make_tokens(
        var_id=tgt_var,
        value=tgt_value,
        prior=torch.zeros(batch_size, tgt_t, bins, device=device),
        mode=torch.full((batch_size, tgt_t), QUERY, device=device),
        mask=torch.ones(batch_size, tgt_t, device=device, dtype=torch.bool),
        x_dim=1,
    )
    return ToyBatch(Batch(vars_, context, target), y[:, :max_context], prior_mu, prior_logsig, mu, log_sigma)


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


def analytic_posterior(
    y_obs: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logsig: torch.Tensor,
    *,
    bins: int,
) -> dict[str, torch.Tensor]:
    """Exact grid posterior for the Gaussian toy.

    The prior is factorized over (`mu`, `log_sigma`) and represented on the same
    grids as ACE's prior tokens. Given observed data, Bayes' rule gives a
    normalized 2D grid posterior. We return the two 1D marginals and moments for
    a compact diagnostic.
    """

    device = y_obs.device
    mu_grid = torch.linspace(MU_RANGE[0], MU_RANGE[1], bins, device=device)
    logsig_grid = torch.linspace(LOGSIG_RANGE[0], LOGSIG_RANGE[1], bins, device=device)
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
    return {
        "mu_grid": mu_grid,
        "logsig_grid": logsig_grid,
        "mu_mean": (pmu * mu_grid).sum(),
        "mu_std": ((pmu * (mu_grid - (pmu * mu_grid).sum()).pow(2)).sum()).sqrt(),
        "logsig_mean": (ps * logsig_grid).sum(),
        "logsig_std": ((ps * (logsig_grid - (ps * logsig_grid).sum()).pow(2)).sum()).sqrt(),
        "pmu": pmu,
        "plogsig": ps,
    }


def normalized_moments(grid: torch.Tensor, log_density: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std of a 1D density evaluated on an evenly spaced grid."""

    p = (log_density - torch.logsumexp(log_density, dim=0)).exp()
    mean = (p * grid).sum()
    std = (p * (grid - mean).pow(2)).sum().sqrt()
    return mean, std


def train(args: argparse.Namespace) -> ACE:
    """Train ACE online on freshly sampled Gaussian-toy batches."""

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
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
    model = ACE(vars_, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(1, args.steps + 1):
        toy = sample_toy_batch(
            vars_,
            batch_size=args.batch_size,
            max_context=args.max_context,
            min_context=args.min_context,
            data_targets=args.data_targets,
            bins=args.bins,
            device=device,
        )
        loss = model.loss(toy.batch, latent_weight=args.latent_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}")
    return model


@torch.no_grad()
def evaluate(model: ACE, args: argparse.Namespace) -> None:
    """Compare ACE's posterior marginals with the analytic grid oracle."""

    device = next(model.parameters()).device
    vars_ = model.variables
    toy = sample_toy_batch(
        vars_,
        batch_size=1,
        max_context=args.max_context,
        min_context=args.max_context,
        data_targets=0,
        bins=args.bins,
        device=device,
    )
    ctx = toy.batch.context
    n = int(ctx.mask[:, : args.max_context].sum().item())
    y_obs = toy.y_context[0, :n]
    true = analytic_posterior(y_obs, toy.prior_mu[0], toy.prior_logsig[0], bins=args.bins)
    mu_grid = true["mu_grid"]
    logsig_grid = true["logsig_grid"]
    mu_logp = query_log_density(model, toy.batch, 1, mu_grid)
    logsig_logp = query_log_density(model, toy.batch, 2, logsig_grid)
    mu_mean, mu_std = normalized_moments(mu_grid, mu_logp)
    logsig_mean, logsig_std = normalized_moments(logsig_grid, logsig_logp)
    print("\nHeld-out Gaussian toy posterior moments")
    print(f"truth mu        {float(toy.mu[0]): .3f}")
    print(f"truth sigma     {float(toy.log_sigma[0].exp()): .3f}")
    print(f"oracle mu       mean {float(true['mu_mean']): .3f}  std {float(true['mu_std']): .3f}")
    print(f"model  mu       mean {float(mu_mean): .3f}  std {float(mu_std): .3f}")
    print(f"oracle log_sig  mean {float(true['logsig_mean']): .3f}  std {float(true['logsig_std']): .3f}")
    print(f"model  log_sig  mean {float(logsig_mean): .3f}  std {float(logsig_std): .3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=64)
    p.add_argument("--max-context", type=int, default=24)
    p.add_argument("--min-context", type=int, default=6)
    p.add_argument("--data-targets", type=int, default=4)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent-weight", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()
    model = train(args)
    evaluate(model, args)


if __name__ == "__main__":
    main()
