// Copies the WASM runtime and the deployed model into public/ so they are served
// locally (no CDN) and precached by the service worker. Run by predev and prebuild.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const models = join(root, "..", "models");

function copy(from: string, toDir: string, name: string) {
  if (!existsSync(from)) throw new Error(`Missing asset: ${from}`);
  mkdirSync(toDir, { recursive: true });
  copyFileSync(from, join(toDir, name));
}

const ort = join(root, "node_modules", "onnxruntime-web", "dist");
for (const file of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  copy(join(ort, file), join(root, "public", "ort"), file);
}
for (const file of ["plant-doctor-int8.onnx", "labels.json"]) {
  copy(join(models, file), join(root, "public", "model"), file);
}
