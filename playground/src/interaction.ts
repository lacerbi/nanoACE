/**
 * Small shared helpers for point-based demos. These keep hit testing, clamping,
 * and OOD reason formatting consistent without coupling the task-specific
 * inference paths.
 */

import type { Plot } from "./plot";

export interface XYPoint {
  x: number;
  y: number;
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}

export function hitPoint(points: XYPoint[], plot: Plot | null, px: number, py: number, radiusPx: number): number | null {
  if (!plot) return null;
  for (let i = 0; i < points.length; i++) {
    const dx = plot.xPx(points[i].x) - px;
    const dy = plot.yPx(points[i].y) - py;
    if (Math.hypot(dx, dy) <= radiusPx) return i;
  }
  return null;
}

export function pointOodReasons(
  points: XYPoint[],
  opts: {
    yIsOod: (y: number) => boolean;
    yReason: string;
    maxPoints: number;
    maxReason?: (n: number) => string;
    minPoints?: number;
    minReason?: (n: number) => string;
  },
): string[] {
  const reasons: string[] = [];
  const nFar = points.filter((p) => opts.yIsOod(p.y)).length;
  if (nFar > 0) reasons.push(`${nFar} point(s) ${opts.yReason}`);
  if (points.length > opts.maxPoints) {
    reasons.push(opts.maxReason ? opts.maxReason(points.length) : `${points.length} points (training used ≤ ${opts.maxPoints})`);
  }
  if (opts.minPoints !== undefined && points.length < opts.minPoints) {
    reasons.push(opts.minReason ? opts.minReason(points.length) : `${points.length} points (training used at least ${opts.minPoints})`);
  }
  return reasons;
}
