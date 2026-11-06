import { describe, expect, it } from "vitest";
import { softmax, topK } from "../src/classifier";

describe("softmax and top-k", () => {
  it("sums to one and survives huge logits", () => {
    const p = softmax([1000, 1001, 999]);
    expect(Array.from(p).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[1]).toBeGreaterThan(p[0]);
  });

  it("returns the k most likely classes in order", () => {
    const top = topK([0.1, 0.5, 0.05, 0.3, 0.05], 3);
    expect(top.map((t) => t.index)).toEqual([1, 3, 0]);
  });
});
