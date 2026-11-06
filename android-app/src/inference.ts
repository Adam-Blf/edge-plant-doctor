import { Asset } from "expo-asset";
import { InferenceSession, Tensor } from "onnxruntime-react-native";
import { SIZE } from "./preprocess";
import { softmax, topK, type Prediction } from "./scores";

export interface Classification {
  top: Prediction[];
  /** Wall-clock milliseconds of session.run plus softmax and top-k, measured on the device. */
  milliseconds: number;
}

const MODEL = require("../assets/model/plant-doctor-int8.onnx");

/**
 * Native ONNX Runtime (CPU execution provider) on the bundled int8 model. The model is
 * a Metro asset, expo-asset hands ORT a local file path (the pattern of the ORT Expo example).
 */
export class Classifier {
  private constructor(private readonly session: InferenceSession) {}

  static async load(): Promise<Classifier> {
    const [asset] = await Asset.loadAsync(MODEL);
    if (!asset?.localUri) throw new Error("Modèle introuvable dans l'application");
    const session = await InferenceSession.create(asset.localUri, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
    });
    return new Classifier(session);
  }

  async classify(input: Float32Array, k = 3): Promise<Classification> {
    const started = performance.now();
    const tensor = new Tensor("float32", input, [1, 3, SIZE, SIZE]);
    const outputs = await this.session.run({ input: tensor });
    const logits = outputs.logits.data as Float32Array;
    const top = topK(softmax(logits), k);
    return { top, milliseconds: performance.now() - started };
  }
}
