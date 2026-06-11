/**
 * Pure inference for the AR-buffer tab (DOM-free, so it can be parity-tested).
 *
 * Three pieces: the context builder (GP-tab token semantics: data points, kernel
 * pin as a discrete VALUE, lengthscale/outputscale pins as zero-spread PRIOR
 * tokens), a static pass through the inherited plain forward (diagonal band +
 * per-grid-point MDN params, from which the independent marginal samples are
 * drawn), and `JointSampler`, a step-driveable wrapper over the buffered
 * incremental decode so the demo can animate the autoregressive reveal.
 */

import { ARBUF } from "../config";
import { PRIOR, QUERY, VALUE, type ACEModel, type TokenSet } from "../ace/model";
import { BufferedACEModel, mdnLogProb, type CtxCache, type DrawState } from "../ace/buffered";
import { Predictions, type MDNParams } from "../ace/predictions";
import { encodeValue } from "../ace/schema";
import { TokenList } from "../ace/tokens";
import { randomOrder, sampleMDN, type RNG } from "../ace/rng";

export interface ArbufSpec {
  points: { x: number; y: number }[];
  pinKernel: number | null;
  pinEll: number | null; // native log-lengthscale when pinned, else null
  pinScale: number | null; // native log-outputscale when pinned, else null
}

/** Context tokens for the buffered model — dense and all-active by construction. */
export function buildContext(model: ACEModel, spec: ArbufSpec): TokenSet {
  const ellMeta = model.variables[1];
  const scaleMeta = model.variables[2];
  const c = new TokenList();
  for (const p of spec.points) c.add(0, VALUE, { x: p.x, value: p.y });
  if (spec.pinKernel !== null) c.add(3, VALUE, { value: spec.pinKernel, valueIndex: spec.pinKernel });
  if (spec.pinEll !== null) {
    const e = encodeValue(ellMeta, spec.pinEll);
    c.add(1, PRIOR, { value: e, prior: [e, 0] });
  }
  if (spec.pinScale !== null) {
    const e = encodeValue(scaleMeta, spec.pinScale);
    c.add(2, PRIOR, { value: e, prior: [e, 0] });
  }
  return c.get();
}

export interface ArbufStatic {
  grid: number[];
  bandMean: number[];
  bandStd: number[];
  params: MDNParams[]; // per-grid-point context-only MDN (independent samples come from these)
}

/** One plain (context-only) forward over the grid: band + per-point MDN params. */
export function arbufStatic(model: ACEModel, context: TokenSet, grid: number[]): ArbufStatic {
  const t = new TokenList();
  for (const x of grid) t.add(0, QUERY, { x });
  const target = t.get();
  const pred = new Predictions(model, model.forward(context, target));
  const bandMean: number[] = [];
  const bandStd: number[] = [];
  const params: MDNParams[] = [];
  for (let i = 0; i < grid.length; i++) {
    bandMean.push(pred.continuousMean(i));
    bandStd.push(Math.sqrt(Math.max(pred.continuousVar(i), 0)));
    params.push(pred.contParams(i));
  }
  return { grid, bandMean, bandStd, params };
}

/** Independent per-point samples from the diagonal marginals — the "no coherence"
 *  reference lines. Returns [nDraws][gridIndex]. */
export function sampleIndependent(stat: ArbufStatic, nDraws: number, rng: RNG): number[][] {
  const out: number[][] = [];
  for (let b = 0; b < nDraws; b++) {
    out.push(stat.params.map((p) => sampleMDN(p, rng)));
  }
  return out;
}

export interface JointSamplerOpts {
  nDraws?: number;
  rng: RNG;
  order?: number[]; // explicit decode order over grid indices (tests); default: random
  teacher?: number[][]; // [nDraws][gridIndex] forced values (tests); default: sample
}

/**
 * Step-driveable coherent joint sampling: one cached context encoding shared by
 * all draw streams; each `step()` decodes one grid location for every draw
 * (predict → sample → append). `values`/`logps` are grid-aligned, NaN until the
 * location is realized — exactly `arbuffer.sample_joint`, one step at a time.
 */
export class JointSampler {
  readonly order: number[];
  readonly values: number[][];
  readonly logps: number[][];
  private pos = 0;
  private draws: DrawState[];
  private teacher: number[][] | null;
  private rng: RNG;

  constructor(
    private model: BufferedACEModel,
    private cache: CtxCache,
    private grid: number[],
    opts: JointSamplerOpts,
  ) {
    const nDraws = opts.nDraws ?? ARBUF.DRAWS;
    this.rng = opts.rng;
    this.order = opts.order ?? randomOrder(grid.length, opts.rng);
    this.teacher = opts.teacher ?? null;
    this.values = Array.from({ length: nDraws }, () => new Array<number>(grid.length).fill(NaN));
    this.logps = Array.from({ length: nDraws }, () => new Array<number>(grid.length).fill(NaN));
    this.draws = Array.from({ length: nDraws }, () => model.newDraw());
  }

  get done(): boolean {
    return this.pos >= this.grid.length;
  }

  get steps(): number {
    return this.pos;
  }

  /** Decode the next grid location for all draws; false once the chain is done. */
  step(): boolean {
    if (this.done) return false;
    const j = this.order[this.pos];
    const x = this.grid[j];
    for (let b = 0; b < this.draws.length; b++) {
      const { params } = this.model.predict(this.cache, this.draws[b], x);
      const y = this.teacher ? this.teacher[b][j] : sampleMDN(params, this.rng);
      this.values[b][j] = y;
      this.logps[b][j] = mdnLogProb(params, y);
      this.model.append(this.cache, this.draws[b], x, y);
    }
    this.pos += 1;
    return true;
  }

  runAll(): void {
    while (this.step()) {
      // drain
    }
  }
}
