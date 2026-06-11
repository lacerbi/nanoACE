/**
 * Seeded RNG + sampling helpers for the AR-buffer demo. Seeded so draws are
 * reproducible (and the tests deterministic); nothing here needs to match
 * PyTorch's RNG — only the *distributions* matter, parity covers the math.
 */

import type { MDNParams } from "./predictions";

export type RNG = () => number;

/** mulberry32: tiny, fast, good-enough 32-bit seeded PRNG. Returns U[0, 1). */
export function mulberry32(seed: number): RNG {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box–Muller (1 - u keeps log() away from 0). */
export function randn(rng: RNG): number {
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Draw from a mixture-density head: categorical over weights, then a Gaussian. */
export function sampleMDN(params: MDNParams, rng: RNG): number {
  const u = rng();
  let acc = 0;
  let comp = params.logW.length - 1;
  for (let i = 0; i < params.logW.length; i++) {
    acc += Math.exp(params.logW[i]);
    if (u < acc) {
      comp = i;
      break;
    }
  }
  return params.loc[comp] + params.scale[comp] * randn(rng);
}

/** Fisher–Yates shuffle of [0, n) under the given RNG. */
export function randomOrder(n: number, rng: RNG): number[] {
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  return order;
}
