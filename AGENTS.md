## Project

nanoACE is a small, readable implementation of the Amortized Conditioning Engine (ACE):
data, interpretable latents, and runtime prior information are all **tokens**; the model
conditions on one token set and predicts distributions over another.

**Read [DEVLOG.md](DEVLOG.md) before any architectural or scope change.** It records the
design decisions and their rationale, and it is the source of truth for what is in scope.
[README.md](README.md) covers setup and the public-facing summary. The paper this is based
on is in [paper/](paper/) as markdown.

## Commands

Windows / PowerShell, using the project virtualenv explicitly:

```powershell
# setup (CUDA wheel pinned to the local stack: torch 2.11.0+cu128)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# run the Gaussian-toy demo (trains, then prints oracle-vs-model posterior moments)
.\.venv\Scripts\python.exe demo.py

# fast smoke test
.\.venv\Scripts\python.exe demo.py --steps 20 --batch-size 32

# force CPU
.\.venv\Scripts\python.exe demo.py --device cpu --steps 20
```

There is no separate test suite, linter, or build step. **Verification = run `demo.py` and
check the printed model posterior moments track the analytic `oracle` moments** (the
Gaussian toy has a closed-form grid posterior; `evaluate()` in `demo.py` is the correctness
oracle). Keep that check loose, not a strict quality gate (see DEVLOG "Open questions").

## Architecture (the cross-file picture)

Everything routes through one idea: **variables as tokens**. The whole model is in
`ace.py`; `demo.py` is one task built on top of it.

- **Data model (`ace.py`).** `Variable` is the static schema (name, `kind` data/latent,
  continuous/discrete + `cardinality`, `transform`, optional prior grid). `Tokens` is a
  padded, field-parallel struct of `[B, T]` tensors (`var_id, x, value, value_index,
prior, mode, mask`). `Batch` = `variables + context: Tokens + target: Tokens`. Device
  movement lives on these objects (`.to`); `cat_tokens` / `Tokens.column` / `with_values`
  are the manipulation primitives.
- **`mode` drives the embedder.** Each token is `VALUE`, `PRIOR`, or `QUERY` (constants in
  `ace.py`). `ACE._embed` builds `var_embed + mode_embed + x_embed + payload`, where the
  payload is selected by mode: prior-histogram MLP for `PRIOR`, the learned `unknown`
  parameter for `QUERY` (so target truth is ignored even when present), else the value
  embedding. Latent tokens zero out `x`. Continuous values go through an MLP; discrete
  values index one shared embedding table via per-variable offsets (`disc_offsets`).
- **Attention (`ACEBlock`).** Separated **context self-attention** then **target→context
  cross-attention**, pre-LN, no directional mask — conditioning direction is structural
  (targets read context, never each other). Padding is handled by `key_padding_mask` from
  the context `mask`. This is deliberate; do not replace it with a single masked
  `(N+M)²` stream (see DEVLOG "Attention").
- **Heads + dispatch (`Predictions`).** `ACE.forward` returns a `Predictions` object
  wrapping a shared continuous **MDN head** (`3*K` outputs → log-weights/means/scales) and
  a shared **categorical head** (`max_cardinality` logits, masked per-token to the
  variable's `cardinality`). `Predictions.log_prob/.mean/.sample` dispatch per token by
  `is_discrete[var_id]`, so callers never branch on type.
- **Loss is type-agnostic.** `ACE.loss` is just `-log_prob` over active target tokens,
  weighted by scalar `data_weight` / `latent_weight` and the target `mask`.
- **Autoregression is a helper, not architecture.** The base model is a _diagonal_
  prediction map (independent 1-D marginals). `sample_ar` builds joint samples by
  predicting one target, sampling it, appending it to context as a `VALUE` token, and
  repeating.

## Conventions and gotchas

- **One global `prior_bins`.** `ACE.__init__` rejects a `Variable` whose `prior_bins`
  differs from the config; ragged per-variable prior grids are out of scope for now.
- **Priors attach to latents only**, and values are expected pre-normalized to ~[-1, 1] at
  generation time — the embedders/heads assume that range.
- **Target tokens may carry truth** in `value`/`value_index` while `mode == QUERY`; the
  embedder ignores it and the loss uses it. For prediction-only calls, pass dummy values
  and simply don't call `.log_prob`.
- **`temp/` is a _different_, much larger research project** (`gp-regression*`), gitignored.
  Mine it for ideas (rectangular attention, CPU float64 GP sampling) but never import its
  fleet-management structure (caching/Slurm/prefetch/resume matrices). See DEVLOG.
- **Currently implemented:** `ace.py` + `demo.py` only. `data.py` / `train.py` (sharded
  generate→save→train) and the GP-1D example are planned in DEVLOG "Layout" but not yet
  built — build in dependency order, model and toy first.
