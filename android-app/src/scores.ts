export interface Prediction {
  index: number;
  probability: number;
}

/** Numerically stable softmax. */
export function softmax(logits: ArrayLike<number>): Float32Array {
  const values = Array.from(logits);
  let max = -Infinity;
  for (const v of values) if (v > max) max = v;
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return Float32Array.from(exps, (v) => v / sum);
}

/** The k most probable classes, best first. Ties keep the lower class index first. */
export function topK(probabilities: ArrayLike<number>, k: number): Prediction[] {
  return Array.from(probabilities, (probability, index) => ({ index, probability }))
    .sort((a, b) => b.probability - a.probability || a.index - b.index)
    .slice(0, k);
}
