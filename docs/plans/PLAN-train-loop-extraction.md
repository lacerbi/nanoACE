# Plan: Shared `train.py` (loop + checkpoint + common args), examples refactored onto it

Created: 2026-06-08
Status: COMPLETE — all phases done, verified, and doublechecked (2026-06-08).
Decisions: cosine LR default (with `--lr-schedule constant` escape); include simple
resume now; final `--save-checkpoint` model-only; config = dataclass + light YAML.
Changeset: NEW `train.py`; modified `requirements.txt`, `gaussian_toy.py`, `gp1d.py`,
`sbi_sir.py`, `bo1d.py`, `DEVLOG.md`, `AGENTS.md`, `README.md`. (The modified
`playground/*` files are concurrent USER/agent edits, NOT part of this work — excluded.)

**Doublecheck (two independent Opus reviewers): no blockers.** Confirmed clean: no
leftover refs to deleted functions, imports trimmed, zero CLI-default drift, control
flow/closure/seed-ordering/cosine-math correct, docs accurate. Three should-fix
footguns found and FIXED + re-verified:
1. Resume `--steps`-mismatch warning was dead (resumable ckpt lacked `config`) → `fit`
   now records `config=asdict(cfg)` in resumable saves and gates the warning on a real
   resume (`start_step > 1`); warning now fires on mismatch, silent on match.
2. `--ckpt-every` with empty `--save-checkpoint` silently no-op'd → now warns.
3. YAML `set_defaults` bypassed `type=`/`choices` (e.g. `lr: 3e-4` → string) →
   `apply_config_file` now coerces via the action `type` and validates `choices`.
   README notes the residual `store_true`-cannot-be-unset-from-CLI asymmetry.

## Summary

Extract the training boilerplate duplicated across `gaussian_toy.py`, `gp1d.py`,
`sbi_sir.py`, and `bo1d.py` into a new core `train.py`: the Adam+clip loop, the
checkpoint save/load, `build_model`, and a shared argparse parent. Then add the two
training features the project says it wants but does not have — **cosine LR** and
**simple resume** — in one place. `data.py` (the sharded saved-pool path) stays
**deferred**: nothing has hit the "online Cholesky is the bottleneck" trigger yet.

This is mostly *deletion*: each example's `train()`/`save_checkpoint`/`build_model`
collapse to calls into `train.py`, while the task-specific science (`variables()`,
the sampler, `evaluate`, `plot_diagnostic`) stays exactly where it is. `main()` is
**not** centralized — keeping each example runnable/readable end-to-end is a guardrail
(see Out of scope).

## Scope

- **In scope**
  - New `train.py` owning: `common_parser()`, `build_model(args, variables, device)`,
    `TrainConfig` (typed training-config dataclass), `fit(...)` (the loop),
    `save_checkpoint(...)`, `load_checkpoint(...)`, `load_train_state(...)`, a YAML
    config loader, and the documented "no prefetch / synchronous online generation"
    comment.
  - **Light YAML config (chosen approach).** `--config run.yaml` layered under explicit
    CLI flags via `set_defaults`; the resolved run config is saved into the checkpoint
    for provenance. One small dep (PyYAML); plain `main()`; **no framework** (Hydra and
    omegaconf were explicitly considered and declined — see Design decisions).
  - Refactor all four examples onto `train.py` (Phase 1, behavior-preserving).
  - Add cosine LR (+ optional warmup) and simple resume (optimizer/scheduler/step state
    as additive checkpoint keys) in `fit` (Phase 2).
  - Docs: DEVLOG entry, AGENTS.md and README.md updates (Phase 3).
- **Out of scope**
  - `data.py` / sharded pools / `iter_batches` / manifests — deferred until a real
    GP/BO pooled run needs it (DEVLOG "Data layer"). `fit` will take a `() -> Batch`
    callable so the pooled path slots in later with no second code path.
  - Centralizing `main()`, or any `Task`/registry/config-framework abstraction
    (DEVLOG: "single dataclass config, no config framework"; "no separate generic
    command-line wrapper").
  - Retraining/re-exporting any checkpoint. Existing `artifacts/*.pt` stay valid and
    loadable; the GP-1D retrain and real SIR/BO runs remain separate follow-ups.
  - New AMP/EMA/weight-decay/grad-accum/`torch.compile` (YAGNI).

## Design decisions (from exploration)

- **Checkpoint backward-compat.** `save_checkpoint` keeps writing `{cfg, seed,
  state_dict}`, adds a `config` key (the resolved run-config dict, for provenance —
  `vars(args)`, all JSON-serializable scalars), and writes `{optimizer, scheduler, step}`
  *only when those are passed*. `load_checkpoint(path, device, variables) -> ACE` reads
  `cfg`/`state_dict` and `.get(...)`s nothing else, so old files load unchanged. All
  added keys are additive; the playground reads neither the dict nor those keys, so it is
  unaffected. ("Model-only" final checkpoint now means *no optimizer/scheduler/step* — it
  still carries `cfg`/`seed`/`state_dict`/`config`.)
- **The one playground constraint.** `playground/export_weights.py` and
  `playground/parity.py` call `module.load_checkpoint(path, device)` with **two args**
  and rely on `module.variables()`. So each example must **retain** a 2-arg
  `load_checkpoint(path, device)` wrapper that forwards to
  `train.load_checkpoint(path, device, variables())`, and keep `variables()` defined.
  No playground file changes.
- **Shared parser via `set_defaults`.** `train.common_parser()` is an
  `ArgumentParser(add_help=False)` parent holding the **21 args common to all four
  examples** — 11 with identical defaults (`--steps, --device, --seed, --lr,
  --latent-weight, --latent-context-prob, --log-every, --no-plot, --save-checkpoint,
  --load-checkpoint, --eval-only`) and 10 whose defaults differ per example
  (`--batch-size, --max-context, --min-context, --data-targets, --d-model, --heads,
  --layers, --hidden, --components, --plot-path`), which the parent gives neutral
  defaults. Each example does `ArgumentParser(parents=[common_parser()])`, then
  `parser.set_defaults(...)` for the 10 differing ones (e.g. bo1d's `d_model=192,
  heads=16, layers=6, hidden=384, components=12, batch_size=64, min_context=1,
  data_targets=24, plot_path="artifacts/bo1d.png"`), then adds its **task-specific**
  args (not present in all four): `--bins` (G/S/B) or `--oracle-bins`+`--oracle-chunk`
  (P), `--eval-points` (P/S/B), `--jitter` (P/B), `--sigma-obs` (S/B),
  `--prior-uniform-mix` (B), `--sigma-f-max` (B), `--scale-check` (B). No
  parent/child name collisions since task-specific args are never in the parent.
- **`fit` takes a thunk.** `sample_batch: Callable[[], Batch]`; each example passes
  `lambda: sample_X(model.variables, ..., <task kwargs>).batch`. The wrapper types
  (`ToyBatch`/`GPBatch`/...) expose `.batch`; `fit` only ever sees a `Batch`.
- **Config: dataclass + light YAML (chosen; resolves old open-question #4).** argparse
  stays the single merge engine; YAML is layered through `set_defaults`, so precedence is
  **parent defaults < per-example `set_defaults` < `--config` YAML < explicit CLI flags**
  — no `argparse.SUPPRESS`/sentinel gymnastics. Flow per example:
  ```python
  parser = ArgumentParser(parents=[common_parser()])
  parser.set_defaults(**EXAMPLE_DEFAULTS)          # per-example (bo1d bigger model, etc.)
  pre, _ = parser.parse_known_args()               # read --config first
  if pre.config:
      data = yaml.safe_load(open(pre.config)) or {}
      _validate_keys(data, parser)                 # reject unknown keys (typo guard)
      parser.set_defaults(**data)                  # YAML overrides example defaults
  args = parser.parse_args()                        # explicit CLI overrides YAML
  ```
  YAML keys are arg **dest** names (underscored, e.g. `latent_context_prob: 0.5`),
  matching `set_defaults`. `fit` consumes a typed `TrainConfig` built from the resolved
  `args` in one place (`TrainConfig.from_args(args)`), keeping the loop decoupled from
  argparse for later programmatic/`data.py` use. Hydra/omegaconf declined: Hydra is the
  "config framework" DEVLOG excludes, adds heavy deps, and its `@hydra.main` CWD change
  would relocate the examples' relative `artifacts/` outputs; omegaconf adds a non-stdlib
  dep + override grammar for little gain over plain YAML here.
- **Seed ordering — from-scratch path is exactly preserved.** main() sets
  `torch.manual_seed(args.seed)` once, then
  `model = load_checkpoint(...) if load else build_model(...)`, then `fit(...)`. On the
  **from-scratch path** this reproduces today's RNG timing bit-for-bit (today seeds
  inside `train()` immediately before `build_model`; nothing draws RNG between the seed
  and model construction, and nothing between construction and the first batch). **This
  requires `fit` to draw no RNG between receiving the model and the first
  `sample_batch()`** (today only `Adam(...)` sits there, which draws nothing — preserve
  that).
- **Load-then-train path RNG timing changes — and that's acceptable.** Today
  `load_checkpoint` runs in `main()` *before* `torch.manual_seed` (the loaded model's
  param-init draws happen pre-seed, so the data stream starts fresh from the seed). The
  refactor seeds *first*, so a loaded model's `ACE(...)` init consumes draws from the
  seeded stream, shifting the training-batch stream. This path is **not** a
  reproducibility guarantee today anyway (no optimizer/step resume exists), `--eval-only`
  never enters `fit`, and the Phase-0 baseline uses only from-scratch runs. Verified
  (independent code read). Recorded so it isn't mistaken for a regression.

## Phases

### Phase 0: Capture behavior baseline (before any edit)
**Goal**: A reference to prove Phase 1 changes nothing.

**Steps**:
1. [x] Run and save stdout (step-1 and final loss) for each, CPU + fixed seed:
   - `gaussian_toy.py --steps 20 --batch-size 32 --device cpu --seed 0 --no-plot`
   - `gp1d.py --steps 20 --batch-size 16 --device cpu --seed 0 --no-plot`
   - `sbi_sir.py --steps 20 --batch-size 16 --device cpu --seed 0 --no-plot`
   - `bo1d.py --steps 20 --batch-size 16 --device cpu --seed 0 --no-plot`
2. [x] Confirm a local `artifacts/<task>.pt` exists for at least one task (for the
   load-contract check); note which exist.

**Baseline (seed 0, CPU, `--log-every 1` — step 1 / step 20 loss anchors):**
- gaussian_toy: **1.5161 / 1.2728**
- gp1d: **0.9664 / 0.9111**
- sbi_sir: **1.4103 / 0.6770**
- bo1d: **0.7851 / 0.2889**
- All four `artifacts/{gaussian_toy,gp1d,sbi_sir,bo1d}.pt` present locally (so the
  optional parity.py check is feasible, but mutates committed fixtures — use git-diff).

**Verification**:
- [x] Four baseline loss pairs recorded.

### Phase 1: Create `train.py` and refactor examples (behavior-preserving)
**Goal**: Remove the duplication; identical training behavior.

**Work**:
- Add **PyYAML** to `requirements.txt` (only new dep).
- `train.py` with `common_parser()` (includes `--config`), `build_model(args, variables,
  device)`, `TrainConfig` (fields: `steps, lr, latent_weight, data_weight=1.0,
  grad_clip=1.0, log_every`) + `TrainConfig.from_args(args)`, a YAML loader +
  `_validate_keys`, `fit(model, sample_batch, cfg) -> ACE` (constant-LR loop, identical
  body to today), `save_checkpoint(path, model, *, seed, config=None)` (writes
  `cfg/seed/state_dict` + optional `config`), `load_checkpoint(path, device, variables)
  -> ACE`. Add the no-prefetch comment.
- Each example: delete local `train`/`build_model`/`save_checkpoint`; rewrite
  `parse_args` to use the parent + per-example `set_defaults` + `--config` YAML layer +
  task args (per the Config design above); keep a 2-arg `load_checkpoint(path, device)`
  wrapper; update `main()` to seed → build/load → `fit(TrainConfig.from_args(args))` →
  evaluate → save (passing `config=vars(args)`) → plot. Preserve bo1d `--scale-check`,
  gaussian's `evaluate(model, bins=...)`, bo1d's `plot_diagnostic(..., eps=...)`.

**Steps**:
1. [x] Add PyYAML to `requirements.txt` (+ installed `PyYAML==6.0.2` in venv).
2. [x] Write `train.py` (constant + cosine + resume; imports OK). NOTE: built full
   feature set in one pass; Phase-1 behavior verified via `--lr-schedule constant`
   (the baseline is constant-LR), cosine/resume verified in Phase 2.
3. [x] Refactor `gp1d.py` first (has the richest arg set + jitter) as the pattern.
4. [x] Apply the same pattern to `gaussian_toy.py`.
5. [x] Apply the same pattern to `sbi_sir.py`.
6. [x] Apply the same pattern to `bo1d.py`.

**Verification** (all PASSED 2026-06-08):
- [x] Re-run the four Phase-0 commands with `--lr-schedule constant`; losses match the
      baseline exactly. gp1d full 20-step trajectory identical; gaussian/sir/bo step
      1+20 identical (1.5161/1.2728, 1.4103/0.6770, 0.7851/0.2889). (Used `constant`
      because the baseline is constant-LR and cosine is the new default — deviation
      noted under Step 2.)
- [x] `bo1d.py --scale-check` still runs and reports token scale.
- [x] **Config precedence:** `--config run.yaml` with `steps: 3` ran 3 steps; adding
      `--steps 4` overrode to 4 (CLI > YAML); an unknown YAML key was rejected
      (`unknown key(s) in ...: ['bogus_key']`).
- [x] **Load-contract (lightweight):** all four `<mod>.load_checkpoint('artifacts/<task>.pt',
      'cpu')` return an `ACE` (2-arg wrapper intact).
- [x] **Stronger check:** `python playground/parity.py` exited 0 and regenerated the 8
      tracked `playground/test/fixtures/*.json` **byte-identically** (no git diff) — the
      load+forward path is unchanged. (Aside: `playground/index.html`,
      `src/gaussian/demo.ts`, `src/gp/demo.ts` show as modified, but those are
      pre-existing USER edits — parity.py only writes JSON fixtures — so they are NOT
      part of this changeset.)

### Phase 2: Cosine LR + simple resume (additive)
**Goal**: Add the missing training features in one place, without breaking compat.

**Work**:
- `fit`: build a cosine schedule (with `warmup` steps) when `cfg.schedule == "cosine"`,
  else constant; `sched.step()` each iteration. Cosine `T_max` = `cfg.steps` (the run's
  **total** budget). Accept `resume_state=None`; when given, restore
  optimizer/scheduler/`step` and continue from `start_step`. Accept optional
  `checkpoint_path` + `ckpt_every` + `seed` to periodically write a **resumable**
  checkpoint via `save_checkpoint(path, model, seed=..., opt=..., sched=..., step=...)`.
- **Resume semantics (state this in code + docs):** resume continues the **same total
  `--steps` budget** the original run targeted; `--steps` on the resuming invocation
  must equal the original (else cosine `T_max` mismatches the restored scheduler state
  and the LR curve is wrong). The torch **RNG state is not checkpointed**, so the
  post-resume *data stream* differs from an uninterrupted run — resume restores
  optimizer/scheduler/step, not the exact batch sequence. This is the DEVLOG "simple
  resume"; saving RNG state for exact-stream resume is a deliberate non-goal here.
- `save_checkpoint(path, model, *, seed, opt=None, sched=None, step=None)`: write the
  three legacy keys always; add `optimizer/scheduler/step` only when passed.
- `load_train_state(path_or_payload, model, opt, sched) -> start_step`: read additive
  keys; tolerate their absence (returns `start_step=1`).
- `common_parser` gains: `--lr-schedule {cosine,constant}`, `--warmup`, `--resume`,
  `--ckpt-every`. `main()` passes `resume_state` when `--resume` is set.
- Keep the final `--save-checkpoint` **model-only** (committed-artifact/playground
  cleanliness); resume state goes only to `--ckpt-every`/`--resume` files.

**Steps**:
1. [x] Add schedule + resume to `fit` and the checkpoint helpers (built into `train.py`
   in one pass; cosine default, `_build_scheduler`, `load_train_state`, additive keys).
2. [x] Wire the new flags into `common_parser` and each `main()` (parent change → all four
   inherit `--lr-schedule/--warmup/--resume/--ckpt-every`; resume-wiring identical).

**Verification** (all PASSED 2026-06-08 via `artifacts/_phase2_test.py`, since removed):
- [x] Resume round-trip, **same total budget**: a 20-step cosine run dropped a resumable
      ckpt at step 10 (`{cfg, optimizer, scheduler, seed, state_dict, step}`, step==10);
      `load_train_state` returned `start_step=11` with scheduler `last_epoch` restored to
      10 (LR continuity, not restarted); resumed `fit` ran 11..20 with no error. (Losses
      not bit-equal to an uninterrupted run — data RNG not checkpointed, by design.)
- [x] `--lr-schedule constant` reproduces Phase-1 baseline losses exactly (verified in
      Phase 1).
- [x] Legacy 3-key `artifacts/gp1d.pt` loads via `load_checkpoint` (2-arg) and via
      `load_train_state` → `start_step=1`, no crash on missing resume keys.
- [x] Final `save_checkpoint(..., config=vars(args))` writes exactly
      `{cfg, seed, state_dict, config}` and **no** `optimizer/scheduler/step` (final-save
      model-only; `config` additive provenance). Smoke runs complete; load-contract holds.

### Phase 3: Documentation
**Goal**: Record the decision and update the cross-file picture.

**Steps**:
1. [x] DEVLOG.md entry (+ link this plan).
2. [x] AGENTS.md updates (architecture intro + `train.py` bullet + Currently implemented).
3. [x] README.md updates (modules bullet + Setup PyYAML + "Shared training options").

**Work**:
- **DEVLOG.md**: new dated entry — what was extracted and why (4× duplication), the
  cosine/resume additions (incl. resume = same-budget, RNG-not-checkpointed), the
  checkpoint backward-compat design (additive keys + 2-arg `load_checkpoint` wrapper for
  the playground), the from-scratch seed-ordering preservation (and the deliberate
  load-then-train timing change), the "no prefetch — synchronous online generation"
  rationale now living in `train.py`, and that `data.py` stays deferred with `fit`
  already accepting a `() -> Batch` source. **Link this plan file** (`docs/plans/
  PLAN-train-loop-extraction.md`), matching how PLAN-bo1d / PLAN-shared-reveal are
  linked from their DEVLOG entries.
- **AGENTS.md**: add `train.py` to "Currently implemented"; in "Architecture
  (cross-file picture)" note that `train.py` owns the loop/checkpoint/common args + the
  light-YAML config while examples own DGP/eval/plot and their own `main()`; note the
  YAML precedence (CLI > `--config` > example defaults) and the new PyYAML dep; note
  `data.py` still planned.
- **README.md**: add a `train.py` bullet to "Implemented modules"; document `--config
  run.yaml` (with a tiny example block) and `--resume`/`--ckpt-every` under the examples;
  note PyYAML in Setup; refresh "Next work" (drop "build train.py", keep
  data.py/retrains).

**Verification** (PASSED 2026-06-08):
- [x] DEVLOG/AGENTS/README mention `train.py` consistently; AGENTS "Currently
      implemented" + architecture updated; README modules/Setup/Examples updated.
- [x] No stale doc claim that `data.py`/`train.py` are both unbuilt (now: `train.py`
      done, `data.py` deferred). No stale per-example `train()`/`save_checkpoint` source
      reference.

## Risks & mitigations
- **Silent behavior drift in Phase 1.** Mitigated by the Phase-0 baseline + exact-loss
  comparison on the from-scratch path, preserved seed ordering, and the "fit draws no RNG
  before the first batch" constraint.
- **Playground breakage.** Mitigated by keeping the **exactly 2-positional-arg**
  `load_checkpoint(path, device)` wrapper + `variables()` per module. Risk is low (no
  dict-key dependency — verified). Note `parity.py` is a *heavy/mutating* check (all four
  artifacts, regenerates committed fixtures), not the everyday guard.
- **`train.py` module-name shadowing.** Generic name, but no other `train` is importable
  in this repo; examples use `import train` / `from train import ...`. Confirm no
  example still defines a local `train` symbol after refactor.
- **Cosine default surprise.** If cosine becomes the default, future retrains differ
  from the constant-LR committed artifacts (artifacts are regenerable, so acceptable) —
  see Open Questions.
- **Resume + cosine `T_max` footgun.** Resuming with a different `--steps` than the
  original silently produces a wrong LR curve. Mitigated by documenting "resume = same
  total budget" and validating the restored step count against `cfg.steps` (warn/raise on
  mismatch).
- **`TrainConfig` drifting from the CLI.** `build_model` reads arch args from `args`
  while `fit` reads a `TrainConfig`; both derive from the same resolved `args` (via
  `TrainConfig.from_args`) in one place per `main()`, so the surfaces can't diverge.
  `build_model` hardcodes `x_dim=1` (all current examples are 1D); use
  `getattr(args, "x_dim", 1)` to leave the door open.
- **Silent YAML typos.** A misspelled YAML key would otherwise inject a junk attribute
  via `set_defaults` and be ignored. Mitigated by `_validate_keys` rejecting any key not
  in the parser's known dests (covered by a Phase-1 verification check). YAML keys are
  dest names (underscored), which is a small gotcha worth documenting in README.

## Open Questions (all RESOLVED 2026-06-08)
- ~~**Default LR schedule**~~ — **RESOLVED:** cosine-by-default with `--lr-schedule
  constant` escape (matches DEVLOG "keep cosine LR"; future retrains differ from the
  constant-LR committed artifacts, which are regenerable).
- ~~**Resume scope**~~ — **RESOLVED:** include the full simple-resume now
  (optimizer+scheduler+step+periodic `--ckpt-every`).
- ~~**Final `--save-checkpoint`**~~ — **RESOLVED:** keep model-only (+ additive `config`
  provenance key); no optimizer/scheduler/step in the final save.
- ~~**`fit` config surface**~~ — **RESOLVED 2026-06-08:** dataclass + light YAML.
  `TrainConfig` (built via `from_args`) feeds `fit`; `--config run.yaml` is layered under
  CLI via `set_defaults`; resolved config saved into the checkpoint. PyYAML added; no
  framework (Hydra/omegaconf declined). See the Config design decision.

---
**Please review. Edit directly if needed, then confirm to proceed.**
