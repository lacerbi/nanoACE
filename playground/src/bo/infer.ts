/**
 * Pure ACE inference for the BO-1D playground. This mirrors the GP tab's shape:
 * editable observations, always-present prior tokens for x_opt/y_opt, optional
 * zero-spread pins, and live predictive + optimum marginal queries.
 */

import { BO } from "../config";
import { ACEModel, PRIOR, QUERY, VALUE } from "../ace/model";
import { Predictions } from "../ace/predictions";
import { encodeValue } from "../ace/schema";
import { TokenList } from "../ace/tokens";
import { linspace, normalize } from "../util";
import { priorFeatures } from "../gaussian/infer";

export interface BOPoint {
  x: number;
  y: number;
}

export interface BOSpec {
  points: BOPoint[];
  xPriorUnit: number;
  xPriorNu: number;
  yPriorUnit: number;
  yPriorNu: number;
  pinXOpt: number | null;
  pinYOpt: number | null;
}

export interface BOGrids {
  bandX: number[];
  xOptGrid: number[];
  yOptGrid: number[];
}

export interface BOResult {
  bandX: number[];
  bandMean: number[];
  bandStd: number[];
  xOptGrid: number[];
  xOptPost: number[] | null;
  yOptGrid: number[];
  yOptPost: number[] | null;
}

export function scaleY(y: number): number {
  const [lo, hi] = BO.Y_RANGE;
  return (2.0 * (y - lo)) / (hi - lo) - 1.0;
}

export function unscaleY(v: number): number {
  const [lo, hi] = BO.Y_RANGE;
  return lo + 0.5 * (v + 1.0) * (hi - lo);
}

export function yOptPriorFeatures(muUnit: number, nu: number): [number, number] {
  const [a, b] = BO.Y_OPT_RANGE;
  const [lo, hi] = BO.Y_RANGE;
  const meanNative = a + muUnit * (b - a);
  const varUnit = (muUnit * (1.0 - muUnit)) / (nu + 1.0);
  const stdNative = Math.sqrt(Math.max(varUnit, 0.0)) * (b - a);
  const meanInternal = (2.0 * (meanNative - lo)) / (hi - lo) - 1.0;
  const spreadInternal = (2.0 * stdNative) / (hi - lo);
  return [meanInternal, spreadInternal];
}

export function defaultBOGrids(model: ACEModel): BOGrids {
  return {
    bandX: linspace(BO.X_DOMAIN[0], BO.X_DOMAIN[1], BO.BAND_POINTS),
    xOptGrid: linspace(model.variables[1].bound_lo, model.variables[1].bound_hi, BO.LATENT_GRID),
    yOptGrid: linspace(BO.Y_OPT_RANGE[0], BO.Y_OPT_RANGE[1], BO.LATENT_GRID),
  };
}

export function boInfer(model: ACEModel, spec: BOSpec, grids: BOGrids = defaultBOGrids(model)): BOResult {
  const xMeta = model.variables[1];
  const yMeta = model.variables[2];

  const c = new TokenList();
  for (const p of spec.points) c.add(0, VALUE, { x: p.x, value: scaleY(p.y) });

  if (spec.pinXOpt !== null) {
    const e = encodeValue(xMeta, spec.pinXOpt);
    c.add(1, PRIOR, { value: e, prior: [e, 0] });
  } else {
    c.add(1, PRIOR, { prior: priorFeatures(spec.xPriorUnit, spec.xPriorNu) });
  }

  if (spec.pinYOpt !== null) {
    const e = encodeValue(yMeta, spec.pinYOpt);
    c.add(2, PRIOR, { value: e, prior: [e, 0] });
  } else {
    c.add(2, PRIOR, { prior: yOptPriorFeatures(spec.yPriorUnit, spec.yPriorNu) });
  }

  const context = c.get();

  const t = new TokenList();
  const bandRange: [number, number] = [0, grids.bandX.length];
  for (const x of grids.bandX) t.add(0, QUERY, { x });

  let xRange: [number, number] | null = null;
  if (spec.pinXOpt === null) {
    xRange = [t.varId.length, t.varId.length + grids.xOptGrid.length];
    for (const x of grids.xOptGrid) t.add(1, QUERY, { value: encodeValue(xMeta, x) });
  }

  let yRange: [number, number] | null = null;
  if (spec.pinYOpt === null) {
    yRange = [t.varId.length, t.varId.length + grids.yOptGrid.length];
    for (const y of grids.yOptGrid) t.add(2, QUERY, { value: encodeValue(yMeta, y) });
  }

  const target = t.get();
  const out = model.forward(context, target);
  const pred = new Predictions(model, out);

  const bandMean: number[] = [];
  const bandStd: number[] = [];
  for (let i = bandRange[0]; i < bandRange[1]; i++) {
    bandMean.push(unscaleY(pred.continuousMean(i)));
    bandStd.push(Math.sqrt(Math.max(pred.continuousVar(i), 0.0)) * (0.5 * (BO.Y_RANGE[1] - BO.Y_RANGE[0])));
  }

  const logpAll = pred.logProb(target);
  const xOptPost = xRange ? normalize(logpAll.slice(xRange[0], xRange[1])) : null;
  const yOptPost = yRange ? normalize(logpAll.slice(yRange[0], yRange[1])) : null;

  return {
    bandX: grids.bandX,
    bandMean,
    bandStd,
    xOptGrid: grids.xOptGrid,
    xOptPost,
    yOptGrid: grids.yOptGrid,
    yOptPost,
  };
}
