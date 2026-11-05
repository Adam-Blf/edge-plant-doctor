import * as ort from "onnxruntime-web/wasm";

export interface Prediction {
  index: number;
  probability: number;
}

export interface Classification {
  top: Prediction[];
  milliseconds: number;
}

/** Numerically stable softmax. */
export function softmax(logits: ArrayLike<number>): Float32Array {
  const max = Math.max(...Array.from(logits));
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return Float32Array.from(exps, (v) => v / sum);
}

export function topK(probabilities: ArrayLike<number>, k: number): Prediction[] {
  return Array.from(probabilities, (probability, index) => ({ index, probability }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, k);
}

/**
 * onnxruntime-web on its plain WebAssembly backend, one thread. GitHub Pages and iOS
 * Safari cannot serve the cross-origin isolation headers that SharedArrayBuffer
 * needs, so the single-threaded build is the one that runs everywhere.
 */
export class Classifier {
  private constructor(private readonly session: ort.InferenceSession) {}

  static async load(base: string): Promise<Classifier> {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = {
      mjs: `${base}ort/ort-wasm-simd-threaded.mjs`,
      wasm: `${base}ort/ort-wasm-simd-threaded.wasm`,
    };
    const session = await ort.InferenceSession.create(`${base}model/plant-doctor-int8.onnx`, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new Classifier(session);
  }

  async classify(input: Float32Array, k = 3): Promise<Classification> {
    const started = performance.now();
    const tensor = new ort.Tensor("float32", input, [1, 3, 224, 224]);
    const outputs = await this.session.run({ input: tensor });
    const logits = outputs.logits.data as Float32Array;
    return { top: topK(softmax(logits), k), milliseconds: performance.now() - started };
  }
}
