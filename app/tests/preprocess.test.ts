import { describe, expect, it } from "vitest";
import {
  areaSpans,
  centerCropSquare,
  normalize,
  preprocess,
  resizeArea,
  SIZE,
} from "../src/preprocess";

function solid(side: number, value: number) {
  return new Uint8Array(side * side * 4).fill(value);
}

describe("area resize", () => {
  it("rows of weights sum to one", () => {
    for (const [src, dst] of [
      [256, 224],
      [896, 224],
      [100, 224],
    ]) {
      for (const span of areaSpans(src, dst)) {
        expect(span.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      }
    }
  });

  it("keeps a constant image constant", () => {
    const out = resizeArea(solid(300, 77), 300);
    expect(out.length).toBe(SIZE * SIZE * 3);
    expect(Math.max(...out.slice(0, 600))).toBeCloseTo(77, 3);
    expect(Math.min(...out.slice(0, 600))).toBeCloseTo(77, 3);
  });

  it("averages a 2x2 block exactly on an integer downscale", () => {
    const side = 448;
    const img = new Uint8Array(side * side * 4);
    for (let i = 0; i < img.length; i += 4) img[i] = (i / 4) % 2 === 0 ? 0 : 200;
    expect(resizeArea(img, side)[0]).toBeCloseTo(100, 4);
  });
});

describe("crop and normalise", () => {
  it("crops a wide image to its centre square", () => {
    const w = 300;
    const h = 100;
    const img = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 100; x < 200; x++) img[(y * w + x) * 4] = 255;
    const { data, side } = centerCropSquare(img, w, h);
    expect(side).toBe(100);
    expect(Math.min(...data.filter((_, i) => i % 4 === 0))).toBe(255);
  });

  it("normalises white with the ImageNet statistics", () => {
    const white = new Float32Array(SIZE * SIZE * 3).fill(255);
    const out = normalize(white);
    expect(out[0]).toBeCloseTo(2.2489, 3);
    expect(out[SIZE * SIZE]).toBeCloseTo(2.4286, 3);
    expect(out[2 * SIZE * SIZE]).toBeCloseTo(2.64, 3);
  });

  it("produces a 1x3x224x224 tensor", () => {
    expect(preprocess(solid(256, 10), 256, 256).length).toBe(3 * SIZE * SIZE);
  });
});
