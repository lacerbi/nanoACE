/**
 * BO demo verification: TS ACE inference must reproduce bo1d.py on the fixed
 * reference case. No oracle is checked here; BO is intentionally the no-oracle
 * playground task.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { ACEModel } from "../ace/model";
import { type Manifest, weightsFromBytes } from "../ace/weights";
import { boInfer } from "./infer";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

interface DemoRef {
  x_context: number[];
  y_context: number[];
  x_prior_unit: number;
  x_prior_nu: number;
  y_prior_unit: number;
  y_prior_nu: number;
  band_x: number[];
  band_mean: number[];
  band_std: number[];
  x_grid: number[];
  x_post: number[];
  y_grid: number[];
  y_post: number[];
}

function maxViolation(a: number[], b: number[], atol: number, rtol: number): number {
  expect(a.length).toBe(b.length);
  let worst = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const slack = Math.abs(a[i] - b[i]) - (atol + rtol * Math.abs(b[i]));
    if (slack > worst) worst = slack;
  }
  return worst;
}

describe("BO demo orchestration vs bo1d.py", () => {
  const manifest = JSON.parse(
    readFileSync(join(ROOT, "public", "models", "bo1d", "manifest.json"), "utf8"),
  ) as Manifest;
  const bytes = readFileSync(join(ROOT, "public", "models", "bo1d", "weights.bin"));
  const model = new ACEModel(weightsFromBytes(manifest, new Uint8Array(bytes)));
  const ref = JSON.parse(readFileSync(join(ROOT, "test", "fixtures", "bo1d.demo.json"), "utf8")) as DemoRef;

  it("matches predictive band and optimum marginals", () => {
    const points = ref.x_context.map((x, i) => ({ x, y: ref.y_context[i] }));
    const res = boInfer(
      model,
      {
        points,
        xPriorUnit: ref.x_prior_unit,
        xPriorNu: ref.x_prior_nu,
        yPriorUnit: ref.y_prior_unit,
        yPriorNu: ref.y_prior_nu,
        pinXOpt: null,
        pinYOpt: null,
      },
      { bandX: ref.band_x, xOptGrid: ref.x_grid, yOptGrid: ref.y_grid },
    );

    expect(maxViolation(res.bandMean, ref.band_mean, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.bandStd, ref.band_std, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.xOptPost!, ref.x_post, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
    expect(maxViolation(res.yOptPost!, ref.y_post, 1e-3, 1e-3)).toBeLessThanOrEqual(0);
  });
});
