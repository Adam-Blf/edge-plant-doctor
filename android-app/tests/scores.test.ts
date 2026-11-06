import { softmax, topK } from "../src/scores";
import { LABELS, percent, title } from "../src/labels";

describe("softmax", () => {
  it("sums to one and keeps the order of the logits", () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[2]).toBeGreaterThan(p[1]);
    expect(p[1]).toBeGreaterThan(p[0]);
    expect(p[0]).toBeCloseTo(0.0900306, 6);
    expect(p[2]).toBeCloseTo(0.6652410, 6);
  });

  it("does not overflow on large logits", () => {
    const p = softmax([1000, 1000, 0]);
    expect(p[0]).toBeCloseTo(0.5, 6);
    expect(Number.isFinite(p[2])).toBe(true);
  });
});

describe("topK", () => {
  it("returns the k best classes, best first", () => {
    const top = topK([0.1, 0.5, 0.05, 0.3, 0.05], 3);
    expect(top.map((t) => t.index)).toEqual([1, 3, 0]);
    expect(top[0].probability).toBe(0.5);
  });

  it("breaks ties by the lower index", () => {
    expect(topK([0.25, 0.25, 0.25, 0.25], 2).map((t) => t.index)).toEqual([0, 1]);
  });
});

describe("labels", () => {
  it("ships the 38 French labels of the model", () => {
    expect(LABELS).toHaveLength(38);
    expect(LABELS.every((l) => l.crop && l.condition)).toBe(true);
    expect(LABELS.some((l) => l.healthy)).toBe(true);
  });

  it("formats a title and a French percentage", () => {
    expect(title({ id: "x", crop: "Pomme", condition: "Tavelure", healthy: false })).toBe("Pomme, tavelure");
    expect(percent(0.973)).toBe("97 %");
    expect(percent(0.0421)).toBe("4,2 %");
  });
});
