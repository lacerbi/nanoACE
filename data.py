"""Offline sharded data pools (generate -> save -> train).

A caller that wants an offline pool provides:

- `draw_fn(n) -> dict[str, Tensor]`: draw `n` stored instances as a struct of arrays;
- `assemble(inst, variables, n_context, reveal_mask, max_context, device) -> Batch`:
  turn stored rows plus read-time split decisions into a training batch;
- `variables()` and `gen_config()`: the token schema and DGP constants recorded in the
  manifest.

`write_pool` writes `draw_fn` outputs into `.pt` shards plus a manifest. `PoolReader`
loads the manifest, reads touched shard rows, recomputes `n_context` and latent reveal
masks from `(seed, position)`, and returns a `(step) -> Batch` callable for `train.fit`.

Design:

- **Cache only the expensive draws.** A pool stores exactly what the caller's
  `draw_instances` produces (a struct-of-arrays per shard), nothing about the
  context/target split or the reveal mask -- those are recomputed at read time, so the
  reveal strategy can change without regenerating the pool.
- **Two functions, one schema.** `write_pool(draw_fn, ...)` generates; `PoolReader` reads
  and returns `Batch`es via the *same* `assemble` the online path uses. `fit` sees the
  identical `(step) -> Batch` interface either way -- no second training code path.
- **Stateless, index-keyed randomness.** The physical-row shuffle and the per-instance
  split decisions are pure functions of `(seed, logical position)` via `ace.mix_int64`
  (no `torch.Generator` state). The logical position is `p = (step - 1) * B + j`, which
  is *batch-size- and steps-independent* (position `p` is the same dataset under any `B`),
  so a pooled run is reproducible and resume-exact from the `step` `fit` already restores.
- **One provenance check.** The manifest carries the `variables()` schema (a hard gate --
  a wrong schema silently misreads the arrays) and a `sha256` of the DGP `gen_config`
  (forceable with `force=True`). This replaces the heavy multi-axis resume-guard matrix.

- **Bounded read-side memory.** `PoolReader` keeps the manifest in memory, lazily loads only
  the shards touched by a batch, caches a bounded number of shards, and asynchronously
  prefetches shards for upcoming batches. Memory scales with shard size plus the cache and
  in-flight prefetch windows, not with the full pool size.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import torch

from ace import Batch, Variable, mix_int64, mix_seed, reveal_mask_from_index

SCHEMA = "nanoace-pool-v1"

# Distinct splitmix salts so the row-shuffle and split streams are decorrelated namespaces.
_SALT = {"shard": 0xA1, "within": 0xB2, "nctx": 0xC3, "reveal": 0xD4}


# --------------------------------------------------------------------------- #
# Manifest helpers (canonical JSON; variables() repr; DGP config hash)
# --------------------------------------------------------------------------- #


def variables_repr(variables: Sequence[Variable]) -> list[dict]:
    """Serializable, order-preserving view of `variables()` for the manifest/guard."""

    return [
        {
            "name": v.name,
            "kind": v.kind,
            "value_type": v.value_type,
            "cardinality": v.cardinality,
            "transform": v.transform,
            "bounds": list(v.bounds) if v.bounds is not None else None,
        }
        for v in variables
    ]


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_hash(gen_config: dict) -> str:
    """16-hex `sha256` of the DGP-only `gen_config` (canonical JSON). Drift => regenerate."""

    return hashlib.sha256(_canonical_json(gen_config).encode("utf-8")).hexdigest()[:16]


def _to_storage(v: torch.Tensor) -> torch.Tensor:
    """Store continuous fields as float32 and integer (categorical) fields as int64."""

    v = v.detach().cpu().contiguous()
    if v.dtype.is_floating_point:
        return v.to(torch.float32)
    return v.to(torch.int64)


# --------------------------------------------------------------------------- #
# Pool generation
# --------------------------------------------------------------------------- #


def _shard_meta(*, cfg_hash: str, seed: int, shard_index: int, start: int, count: int) -> dict:
    return {"schema": SCHEMA, "config_hash": cfg_hash, "seed": int(seed),
            "shard_index": int(shard_index), "start": int(start), "count": int(count)}


def _valid_shard(path: Path, meta: dict) -> bool:
    if not path.exists():
        return False
    try:
        shard = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return shard.get("__meta__") == meta


def write_pool(
    draw_fn: Callable[[int], dict],
    out: str | Path,
    *,
    pool_size: int,
    shard_size: int,
    gen_config: dict,
    variables: Sequence[Variable],
    seed: int,
    force: bool = False,
    log=print,
) -> dict:
    """Generate a sharded finite pool of drawn instances.

    `draw_fn(n) -> dict[str, Tensor]` returns CPU-native struct-of-arrays for `n`
    instances. Shard `i`
    is produced after `torch.manual_seed(mix_seed(seed, i))`, so a partial build resumes
    deterministically (valid existing shards are skipped). Shards are written atomically
    (temp -> rename); the manifest is written **last**, so "manifest exists => pool complete".
    """

    if pool_size <= 0 or shard_size <= 0:
        raise ValueError("pool_size and shard_size must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} exists; pass force=True to rebuild")

    cfg_hash = config_hash(gen_config)
    n_shards = (pool_size + shard_size - 1) // shard_size
    shards, fields_meta = [], None
    for i, start in enumerate(range(0, pool_size, shard_size)):
        count = min(shard_size, pool_size - start)
        fname = f"shard_{i:05d}.pt"
        path = out / fname
        meta = _shard_meta(cfg_hash=cfg_hash, seed=seed, shard_index=i, start=start, count=count)
        if _valid_shard(path, meta) and not force:
            log(f"[skip] {fname} ({count} instances)")
        else:
            torch.manual_seed(mix_seed(seed, i))
            inst = draw_fn(count)
            shard = {k: _to_storage(v) for k, v in inst.items()}
            shard["__meta__"] = meta
            tmp = path.with_suffix(".pt.tmp")
            torch.save(shard, tmp)
            tmp.replace(path)
            log(f"[write] {fname} ({count} instances) [{i + 1}/{n_shards}]")
        if fields_meta is None:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            fields_meta = [
                {"name": k, "shape": list(shard[k].shape[1:]), "dtype": str(shard[k].dtype).replace("torch.", "")}
                for k in shard
                if k != "__meta__"
            ]
        shards.append({"file": fname, "start": start, "count": count})

    manifest = {
        "schema": SCHEMA,
        "pool_size": int(pool_size),
        "shard_size": int(shard_size),
        "seed": int(seed),
        "gen_config": gen_config,
        "config_hash": cfg_hash,
        "variables": variables_repr(variables),
        "fields": fields_meta,
        "shards": shards,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    log(f"[done] manifest -> {manifest_path}")
    return manifest


# --------------------------------------------------------------------------- #
# Pool reading
# --------------------------------------------------------------------------- #


class PoolReader:
    """Read a pool as a `sample_batch(step) -> Batch` callable for `fit`.

    Validates the manifest on construction: schema and `variables()` are hard gates (a wrong
    token schema would silently misread the cached arrays); a `gen_config` config-hash
    mismatch is refused unless `force=True` (a knowing reuse under changed DGP constants).
    `max_context < N_TOTAL` is required (at least one target). Splits and the "both" shuffle
    are stateless functions of `(seed, p)` with `p = (step - 1) * B + j`. Shards are loaded
    lazily through a bounded LRU cache, and a one-thread prefetcher can queue upcoming
    batch shards because the schedule is deterministic.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        assemble: Callable[..., Batch],
        variables: Sequence[Variable],
        gen_config: dict,
        batch_size: int,
        seed: int,
        max_context: int,
        min_context: int,
        latent_context_prob: float,
        device: torch.device | str,
        force: bool = False,
        cache_shards: int = 4,
        prefetch_batches: int = 1,
    ):
        self.dir = Path(path)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError(f"pool schema {self.manifest.get('schema')!r} != {SCHEMA!r}; regenerate the pool")
        if self.manifest.get("variables") != variables_repr(variables):
            raise ValueError(
                "pool variables() mismatch: the cached arrays would be misread under this schema. "
                "Regenerate the pool (NOT overridable by force)."
            )
        want_hash = config_hash(gen_config)
        if self.manifest.get("config_hash") != want_hash:
            msg = (
                f"pool DGP config-hash mismatch (manifest {self.manifest.get('config_hash')}, "
                f"current {want_hash}); the cached data was generated under different DGP constants."
            )
            if not force:
                raise ValueError(msg + " Regenerate the pool, or pass --pool-force to reuse it anyway.")
            print("warning: " + msg + " Reusing it because --pool-force was given.")

        self.n_total = int(self.manifest["gen_config"]["N_TOTAL"])
        if not max_context < self.n_total:
            raise ValueError(f"max_context ({max_context}) must be < N_TOTAL ({self.n_total}) to leave >=1 target")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 1 <= min_context <= max_context:
            raise ValueError(f"need 1 <= min_context ({min_context}) <= max_context ({max_context})")
        if cache_shards < 1:
            raise ValueError(f"cache_shards must be >= 1, got {cache_shards}")
        if prefetch_batches < 0:
            raise ValueError(f"prefetch_batches must be >= 0, got {prefetch_batches}")

        self.assemble = assemble
        self.variables = list(variables)
        self.B = int(batch_size)
        self.seed = int(seed)
        self.max_context = int(max_context)
        self.min_context = int(min_context)
        self.q = 1.0 - float(latent_context_prob)
        self.device = device
        self.pool_size = int(self.manifest["pool_size"])
        self.n_latents = sum(1 for v in self.variables if v.kind == "latent")

        self.shards = list(self.manifest["shards"])
        self.shard_starts = [int(e["start"]) for e in self.shards]
        self.shard_counts = [int(e["count"]) for e in self.shards]
        if sum(self.shard_counts) != self.pool_size:
            raise ValueError("pool manifest shard counts do not sum to pool_size")
        self._shard_counts_t = torch.tensor(self.shard_counts, dtype=torch.int64)
        self.field_names = [f["name"] for f in self.manifest["fields"]]
        if not self.field_names:
            raise ValueError("pool manifest has no fields")
        self._field_meta = {f["name"]: f for f in self.manifest["fields"]}

        self.cache_shards = int(cache_shards)
        self.prefetch_batches = int(prefetch_batches)
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._futures: dict[int, Future[dict[str, torch.Tensor]]] = {}
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="nanoace-pool")
            if self.prefetch_batches > 0 else None
        )
        self._closed = False
        self._layout_pass: int | None = None
        self._layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._within_cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
        self._prefetch_from(1)

    def _key(self, name: str, *ints: int) -> int:
        """Scalar splitmix-style key for `(seed, name, *ints)`, in `[0, 2**62)`."""

        h = (self.seed + 1) * 0x9E3779B97F4A7C15
        h ^= (_SALT[name] + 1) * 0xBF58476D1CE4E5B9
        for i, x in enumerate(ints):
            h += (int(x) + 1) * (0x94D049BB133111EB + i)
        return h & ((1 << 62) - 1)

    def _index_perm(self, n: int, key: int) -> torch.Tensor:
        return mix_int64(torch.arange(n, dtype=torch.int64) + key).argsort()

    def _pass_layout(self, p: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shard order plus cumulative bounds for one pass, without materializing all rows."""

        if self._layout_pass == p and self._layout is not None:
            return self._layout
        shard_order = self._index_perm(len(self.shards), self._key("shard", p))
        counts = self._shard_counts_t.index_select(0, shard_order)
        ends = torch.cumsum(counts, dim=0)
        starts = ends - counts
        self._layout_pass, self._layout = p, (shard_order, starts, ends)
        return self._layout

    def _within_perm(self, p: int, shard_idx: int) -> torch.Tensor:
        """Within-shard row order for one `(pass, shard)` pair."""

        key = (int(p), int(shard_idx))
        if key in self._within_cache:
            self._within_cache.move_to_end(key)
            return self._within_cache[key]
        perm = self._index_perm(self.shard_counts[shard_idx], self._key("within", p, shard_idx))
        self._within_cache[key] = perm
        max_cached = max(1, self.cache_shards * 2)
        while len(self._within_cache) > max_cached:
            self._within_cache.popitem(last=False)
        return perm

    def _physical_locs(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map absolute logical positions to `(shard_index, local_row)` under the "both" shuffle."""

        pass_idx = pos // self.pool_size
        pass_pos = pos % self.pool_size
        out_shard = torch.empty_like(pos)
        out_local = torch.empty_like(pos)
        for p in torch.unique(pass_idx).tolist():
            m = pass_idx == p
            pp = pass_pos[m]
            shard_order, starts, ends = self._pass_layout(int(p))
            order_pos = torch.searchsorted(ends, pp, right=True)
            shard_idx = shard_order.index_select(0, order_pos)
            within_pos = pp - starts.index_select(0, order_pos)
            local = torch.empty_like(within_pos)
            for s in torch.unique(shard_idx).tolist():
                sm = shard_idx == s
                local[sm] = self._within_perm(int(p), int(s)).index_select(0, within_pos[sm])
            out_shard[m] = shard_idx
            out_local[m] = local
        return out_shard, out_local

    def _expected_shard_meta(self, shard_idx: int) -> dict:
        entry = self.shards[shard_idx]
        return _shard_meta(
            cfg_hash=self.manifest["config_hash"],
            seed=int(self.manifest["seed"]),
            shard_index=shard_idx,
            start=int(entry["start"]),
            count=int(entry["count"]),
        )

    def _load_shard_from_disk(self, shard_idx: int) -> dict[str, torch.Tensor]:
        entry = self.shards[shard_idx]
        path = self.dir / entry["file"]
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard.get("__meta__") != self._expected_shard_meta(shard_idx):
            raise ValueError(f"pool shard metadata mismatch in {path}; regenerate the pool")
        got = {k for k in shard if k != "__meta__"}
        want = set(self.field_names)
        if got != want:
            raise ValueError(f"pool shard field mismatch in {path}: got {sorted(got)}, want {sorted(want)}")
        count = self.shard_counts[shard_idx]
        for name in self.field_names:
            tensor = shard[name]
            meta = self._field_meta[name]
            want_shape = [count] + list(meta["shape"])
            got_dtype = str(tensor.dtype).replace("torch.", "")
            if list(tensor.shape) != want_shape or got_dtype != meta["dtype"]:
                raise ValueError(
                    f"pool shard field {name!r} mismatch in {path}: "
                    f"shape {list(tensor.shape)} dtype {got_dtype}, want {want_shape} {meta['dtype']}"
                )
        return {k: shard[k].contiguous() for k in self.field_names}

    def _remember_shard(self, shard_idx: int, shard: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._cache[shard_idx] = shard
        self._cache.move_to_end(shard_idx)
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return shard

    def _get_shard(self, shard_idx: int) -> dict[str, torch.Tensor]:
        if shard_idx in self._cache:
            self._cache.move_to_end(shard_idx)
            return self._cache[shard_idx]
        future = self._futures.pop(shard_idx, None)
        shard = future.result() if future is not None else self._load_shard_from_disk(shard_idx)
        return self._remember_shard(shard_idx, shard)

    def _positions_for_step(self, step: int) -> torch.Tensor:
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        start = (int(step) - 1) * self.B
        return torch.arange(start, start + self.B, dtype=torch.int64)

    def _prefetch_step(self, step: int) -> None:
        if self._executor is None or self._closed:
            return
        shard_idx, _ = self._physical_locs(self._positions_for_step(step))
        for s in dict.fromkeys(int(x) for x in shard_idx.tolist()):
            if s not in self._cache and s not in self._futures:
                self._futures[s] = self._executor.submit(self._load_shard_from_disk, s)

    def _prefetch_from(self, step: int) -> None:
        for future_step in range(step, step + self.prefetch_batches):
            self._prefetch_step(future_step)

    def _gather(self, shard_idx: torch.Tensor, local_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        inst: dict[str, torch.Tensor] = {}
        for s in torch.unique(shard_idx).tolist():
            m = shard_idx == s
            rows = local_idx[m]
            out_rows = torch.nonzero(m, as_tuple=False).squeeze(1)
            shard = self._get_shard(int(s))
            for name in self.field_names:
                vals = shard[name].index_select(0, rows)
                if name not in inst:
                    inst[name] = torch.empty((shard_idx.numel(),) + tuple(vals.shape[1:]), dtype=vals.dtype)
                inst[name].index_copy_(0, out_rows, vals)
        return inst

    def _n_context(self, pos: torch.Tensor) -> torch.Tensor:
        span = self.max_context - self.min_context + 1
        mixed = mix_int64(pos + self._key("nctx")) & ((1 << 62) - 1)
        return self.min_context + (mixed % span)

    def close(self) -> None:
        if self._executor is not None and not self._closed:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._closed = True

    def __del__(self) -> None:
        self.close()

    def __call__(self, step: int) -> Batch:
        pos = self._positions_for_step(int(step))
        shard_idx, local_idx = self._physical_locs(pos)
        inst = self._gather(shard_idx, local_idx)
        self._prefetch_from(int(step) + 1)
        n_context = self._n_context(pos).to(self.device)
        reveal = reveal_mask_from_index(pos + self._key("reveal"), self.n_latents, self.q).to(self.device)
        return self.assemble(
            inst,
            variables=self.variables,
            n_context=n_context,
            reveal_mask=reveal,
            max_context=self.max_context,
            device=self.device,
        )


# --------------------------------------------------------------------------- #
# Build CLI: python data.py <task> --out DIR --pool-size N [...]
# --------------------------------------------------------------------------- #


def _task_module(name: str):
    if name == "gp1d":
        import gp1d as ex
    elif name == "bo1d":
        import bo1d as ex
    else:
        raise SystemExit(f"unknown task {name!r}; choose gp1d or bo1d")
    return ex


def main() -> None:
    p = argparse.ArgumentParser(description="Build a sharded finite training-data pool (generate -> save).")
    p.add_argument("task", choices=("gp1d", "bo1d"))
    p.add_argument("--out", required=True, help="output pool directory")
    p.add_argument("--pool-size", type=int, required=True, help="number of instances in the finite pool")
    p.add_argument("--shard-size", type=int, default=8192, help="instances per shard")
    p.add_argument("--seed", type=int, default=0, help="build seed; shard i uses mix_seed(seed, i)")
    p.add_argument("--force", action="store_true", help="overwrite an existing complete pool")
    args = p.parse_args()

    ex = _task_module(args.task)
    print(f"== build {args.task} pool ==  out={args.out}  pool_size={args.pool_size}  shard_size={args.shard_size}")
    write_pool(
        ex.draw_pool,
        args.out,
        pool_size=args.pool_size,
        shard_size=args.shard_size,
        gen_config=ex.gen_config(),
        variables=ex.variables(),
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
