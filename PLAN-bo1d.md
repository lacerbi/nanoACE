# Plan: 1D Bayesian optimization example (`bo1d.py`)

Created: 2026-06-07
Status: PLANNED

## Summary

A fourth executable example: 1D Bayesian optimization. The two latents are the
**location** `x_opt` and **value** `y_opt` of the global optimum (minimum) of a
black-box function. ACE amortizes `p(x_opt | D)` and `p(y_opt | D)` directly --
the distributions that normally make BO need bespoke acquisition machinery -- and
accepts a **runtime Beta prior over the optimum location** (the paper's πBO /
ACEP-TS story).

This is a deliberate mix of the existing examples:

- **from `gp1d.py`**: online GP function sampling (CPU float64 Cholesky), `(x, y)`
  data tokens, `x_dim = 1`, sampled kernel/hyperparameters;
- **from `gaussian_toy.py` / `sbi_sir.py`**: ACEP Beta prior tokens over bounded
  continuous latents, the `latent_context_prob` reveal mechanism, and the fixed
  diagnostic plot pattern;
- **new**: latents that are properties of the *specific sampled function* (the
  optimum) rather than of the function class; the optimum-planting DGP; an
  **ε-contamination** ("robust prior") mechanism; and a **no grid oracle**
  verification recipe.

Paper reference: Appendix C.3.1 (BO dataset generation), C.1 (GP sampling), B.1
(prior injection). The paper is inspiration, not a constraint (see DEVLOG).

## Scope

- **In scope**:
  - `bo1d.py`: schema, online DGP, training loop, fixed diagnostic, checkpoint
    helpers, plot.
  - ε-contamination helpers in `ace_prior.py`: contaminated truth-sampling and a
    mixture log-prior for plot overlays.
  - Three-column prior diagnostic (uniform / correct-informative /
    wrong-informative) plus a posterior-predictive panel.
  - Docs: this plan, a DEVLOG entry, `README.md` run section, `AGENTS.md`
    "currently implemented" list, and the no-oracle verification recipe.
  - Short verification runs and a token-scale histogram sanity check.
- **Out of scope**:
  - **No grid oracle.** The `|·|` fold in the DGP destroys Gaussianity, so there
    is no closed-form posterior. Verification is qualitative against known truth
    (see "Verification").
  - **No BO loop.** No iterative acquisition / Thompson-sampling rollout. We show
    the conditional `p(x_opt | y_opt = v, D)` to gesture at TS, nothing more.
  - **No new core (`ace.py`) machinery.** Both latents are bounded continuous and
    reuse the existing PRIOR-token path unchanged.
  - **No discrete latent.** Kernel/hyperparameters are sampled nuisance, not
    predicted (`gp1d.py` already exercises the discrete path).
  - Heavy convergence tuning beyond a working example artifact.

## Schema

```python
[
    Variable("y", "data", "continuous"),
    Variable("x_opt", "latent", "continuous", bounds=(-1.0, 1.0)),
    Variable("y_opt", "latent", "continuous", bounds=Y_RANGE),
]
```

- `x_opt`: identity transform, bounds `(-1, 1)`; `encode_value` is then ~identity.
  This is the headline Beta-prior target (πBO).
- `y_opt`: identity transform, bounds `Y_RANGE` covering the **full** native y
  working range (see "Scaling"), so `encode_value(y_opt)` and the data-`y`
  scaling share one affine.
- `y`: continuous data variable, scaled by the same affine at the token boundary.

`p(x_opt | D)` is genuinely multimodal under sparse data (competing basins), so
this is the first example that exercises the MDN's multi-component capacity.

## Data-generating process (full, per sample)

Faithful to Appendix C.3.1, adapted to 1D, with the optimum **value** leveled by
our prior instead of the paper's `U[-5, 5]` offset.

1. **Nuisance hyperparameters** (not latents): kernel from
   {RBF, Matérn-1/2, Matérn-3/2, Matérn-5/2} with weights `[0.35, 0.1, 0.2, 0.35]`;
   lengthscale `ℓ ~ N(1/3, 0.75)` truncated `[0.05, 2]`; output scale
   `σ_f ~ U(0.1, σ_f_max)`. `σ_f_max` is tamed (provisionally `0.5`) so the bump
   above the optimum stays near the `[-1, 1]` token convention.
2. **Latents**:
   - `x_opt` from the ε-contaminated Beta prior on `[-1, 1]`.
   - Natural optimum depth `d`: minimum of `N` draws from `N(0, σ_f²)` with
     `N ≈ ceil(2 / ℓ)` (uncorrelated samples across the domain); with `p = 0.1`,
     `d -= Exp(1)` ("unexpectedly low optimum"). `d < 0`.
   - `y_opt` from the ε-contaminated Beta prior on `Y_RANGE` (the leveling shift).
3. **GP draw conditioned on the optimum geometry**: sample a joint GP prior draw
   `g` over `{x_opt} ∪ x_data`, then condition on `g(x_opt) = d` with the exact
   linear correction
   `g_c(x) = g(x) − (k(x, x_opt) / k(x_opt, x_opt)) · (g(x_opt) − d)`.
   The `−d` term adds a smooth depth-`d` bowl at `x_opt`, so deeper natural
   optima give smoother local geometry. One Cholesky over the joint set; context
   and targets come from the same draw (simpler than the paper's
   independent-target speedup, cheap at our `N`).
4. **Fold + envelope + level**:
   `f(x) = |g_c(x) − d| + (1/5) · (x − x_opt)² + y_opt`.
   At `x_opt`: `|d − d| + 0 + y_opt = y_opt`; everywhere else strictly greater.
   So `x_opt` is the exact, unique global minimum with value `y_opt`. The `|·|`
   fold is what produces the kinked, erratic Fig. S12 samples -- faithful, and
   exactly why there is no closed-form oracle.
5. **Tokenize**: scale `f` values by the shared affine; emit data tokens
   `(x_i, scaled f_i)`, one Beta PRIOR token per latent, QUERY targets for the
   held-out data points and the (unrevealed) latents.

Envelope constant `1/5` is hardcoded (paper value); revisit only if functions
look too bowl-dominated.

## ε-contamination (robust prior)

The effective generative prior over each latent is
`(1 − ε) · Beta(α, β) + ε · Uniform`, a classic ε-contamination / robust-Bayes
prior. ε is a fixed hyperparameter (provisionally `0.1`), exposed as
`--prior-uniform-mix`.

- **Token**: still encodes only the user's raw `Beta(α, β)` (unchanged 2-feature
  representation). The ε floor is a global constant, so the model learns to
  discount every injected prior automatically -- it never needs to be in the
  token.
- **Truth-sampling (DGP)**: draw from the mixture (flip an ε coin: uniform else
  `draw_from_beta`). This is where the model learns the discount.
- Applied to **both** latents.

New helpers in `ace_prior.py`:

- `sample_contaminated(mu_unit, nu, lo, hi, eps)` -> native draw from the mixture.
- `mixture_logprior_on_grid(grid_native, mu_unit, nu, lo, hi, eps)` ->
  `logsumexp([log(1−eps) + logBeta, log(eps) − log(hi−lo)])`, for overlaying the
  effective prior on diagnostic panels.

Because there is no oracle, these are used for plotting/coupling only, not for a
posterior computation.

## Scaling (expected to need tuning)

`y_opt` and data `y` are the **same physical quantity**, so they share one affine:

- `Y_RANGE` is the full native y working range (min through max), used both as the
  `y_opt` latent `bounds` and as the data-`y` scaling. `y_opt` posterior then
  lives in the lower part of `[-1, 1]`; data fills the rest. The model sees both
  on one ruler, so `y_opt ≤ all y` is legible.
- Spread budget: envelope `≤ 0.8`; `|g_c − d|` scales with `σ_f`; native
  `y ∈ [y_opt, y_opt + bump]`.
- Provisional numbers: `σ_f ~ U(0.1, 0.5)` (bump ≲ 1.5), native `y_opt ∈ [-1, 0]`,
  native `y ∈ ~[-1, 1.5]`, `Y_RANGE = (-1.25, 1.75)`.

**First implementation step is a scale check**: sample a batch of functions and
confirm the token-value histogram sits in `[-1, 1]` before any training. Adjust
`σ_f_max`, the envelope constant, and `Y_RANGE` as needed.

## Diagnostic and verification (no oracle)

Fixed evaluation case (seeded), evaluated under three prior settings on `x_opt`:

1. **uniform** prior,
2. **correct informative** prior (tightens toward true `x_opt`),
3. **wrong informative** prior (mass on the wrong basin) -- ACE still recovers the
   true optimum thanks to the ε floor. This panel is the whole reason for the
   mechanism and is a permanent part of the diagnostic.

Plot (provisional layout):

- top: the true function, context points, true `(x_opt, y_opt)` marked, and the
  ACE posterior-predictive band;
- middle: `p(x_opt | D)` marginal per prior column, with the effective
  (contaminated) prior overlaid and true `x_opt` marked;
- bottom: `p(y_opt | D)` marginal, with the conditional `p(x_opt | y_opt = v, D)`
  shown to gesture at Thompson sampling.

`latent_context_prob` reveals `x_opt` *or* `y_opt` as a zero-spread known token
during training, so the AR/conditional queries are in-distribution. Reuses
`diagnostics.py` helpers (`query_log_density`, `conditional_log_density`).

**Verification recipe** (since the project convention of a grid oracle does not
apply): short run that completes, the token-scale histogram check, and visual
inspection of the fixed diagnostic -- ACE marginals should concentrate near the
marked truth, the correct prior should tighten `p(x_opt | D)`, and the wrong
prior should be overridden by the data. This recipe goes into `AGENTS.md` and
`DEVLOG.md`.

## CLI / artifacts

Mirror `gp1d.py` / `sbi_sir.py`: `--steps`, `--batch-size`, `--device`, `--seed`,
context/target sizes, model size, `--lr`, `--latent-weight`,
`--latent-context-prob`, `--prior-uniform-mix`, `--sigma-f-max`, `--jitter`,
`--eval-points`, plot/checkpoint/eval-only flags. Artifacts: `artifacts/bo1d.pt`,
`artifacts/bo1d.png`.

## Implementation order

1. Schema + `make_tokens` + scaling constants; **scale-check** a sampled batch.
2. DGP (`sample_bo_batch`): hyperparameters, contaminated latent draws, GP draw +
   optimum conditioning, fold/envelope/level, tokenization.
3. ε-contamination helpers in `ace_prior.py`.
4. Training loop + checkpoint helpers (copy the established pattern).
5. Fixed eval case + three-prior diagnostic + plot.
6. Short verification run; tune scales; docs (DEVLOG entry, README, AGENTS).

## Open questions / risks

- **Spurious global minima**: with a weak bump the GP `|·|` term could create a
  lower point off `x_opt`. The envelope + conditioning depth make this rare;
  mitigate by tuning `σ_f_max` / envelope constant. Optionally assert the labeled
  `x_opt` is the dense-grid argmin in the fixed diagnostic.
- **Scale numbers are provisional** and expected to change after the histogram
  check (flagged by the user).
- **Scope**: this is example #4; DEVLOG's "nano ships exactly two" is already
  stretched by SIR. BO earns its place (instance-level latents + the optimum
  posterior headline + robust prior injection), recorded as a deliberate
  decision in the DEVLOG entry.
