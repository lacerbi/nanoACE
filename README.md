# nanoACE

nanoACE is a small, readable implementation of the core ideas behind the
Amortized Conditioning Engine (ACE): treat data, interpretable latents, and
runtime prior information as tokens; condition on one token set; predict
distributions over another token set.

The goal is legible source that a human or coding agent can read end to end and
extend. It is not a packaged ACE runtime, not a benchmark suite, and not a clone
of the original research code.

## Reference

This project is based on:

```bibtex
@article{chang2025amortized,
  title={Amortized Probabilistic Conditioning for Optimization, Simulation and Inference},
  author={Chang, Paul E and Loka, Nasrulloh and Huang, Daolang and Remes, Ulpu and Kaski, Samuel and Acerbi, Luigi},
  journal={28th Int. Conf. on Artificial Intelligence & Statistics (AISTATS 2025)},
  year={2025}
}
```

Local paper markdown is in [paper/](paper/). The upstream paper page links to
the paper, markdown, and original ACE code:
[chang2025amortized_overview.md](paper/chang2025amortized_overview.md).

## Current Status

The first working slice is implemented:

- [ace.py](ace.py): core `Variable`, `Tokens`, `Batch`, ACE transformer, shared
  continuous MDN head, shared masked categorical head, prediction object, loss,
  and autoregressive sampling helper.
- [demo.py](demo.py): Gaussian toy with two continuous latents, runtime prior
  injection, online training, and an analytic grid posterior diagnostic.
- [DEVLOG.md](DEVLOG.md): design decisions and rationale. Read this before
  changing architecture or scope.

Planned next example: GP-1D regression with continuous kernel hyperparameter
latents plus discrete kernel selection.

## Setup

Use a local virtual environment. The pinned stack matches the local GPU setup
used in related experiments:

- `torch==2.11.0+cu128`
- PyTorch CUDA runtime 12.8
- NVIDIA RTX 4060 Laptop GPU

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the demo:

```powershell
.\.venv\Scripts\python.exe demo.py
```

For a quick smoke test:

```powershell
.\.venv\Scripts\python.exe demo.py --steps 20 --batch-size 32
```

To force CPU:

```powershell
.\.venv\Scripts\python.exe demo.py --device cpu --steps 20
```

## Design Notes

nanoACE keeps the ACE conditioning semantics, but the paper math is a starting
point rather than a constraint. The invariants are:

- variables are tokens;
- data values, latent values, and latent priors can appear in context;
- target tokens request predictive distributions;
- the training path is type-agnostic through `dist.log_prob`;
- the model uses separated context self-attention and target-to-context
  cross-attention.

The internal token representation is intentionally explicit:

```python
Batch(
    variables: list[Variable],
    context: Tokens,
    target: Tokens,
)

Tokens(
    var_id: LongTensor[B, T],
    x: FloatTensor[B, T, x_dim],
    value: FloatTensor[B, T],
    value_index: LongTensor[B, T],
    prior: FloatTensor[B, T, n_bins],
    mode: LongTensor[B, T],   # VALUE | PRIOR | QUERY
    mask: BoolTensor[B, T],
)
```

## Agent Notes

Before making architectural changes, read [DEVLOG.md](DEVLOG.md). The project
values local, readable code over benchmark machinery. In particular:

- keep `ace.py` as the main readable implementation file;
- avoid importing fleet-management ideas from `temp/`;
- keep examples small and diagnostic;
- prefer changing the implementation when a simpler or more robust tweak serves
  ACE's conditioning interface better than paper fidelity.

Generated directories such as `.venv/`, `__pycache__/`, and `temp/` are ignored.

