// Reads width, height (uint32 LE) then RGBA bytes, writes the float32 model input.
// Used by tests/test_web_parity.py to compare the web pipeline with the Python one.
import { readFileSync, writeFileSync } from "node:fs";
import { preprocess } from "../src/preprocess.ts";

const [input, output] = process.argv.slice(2);
const bytes = readFileSync(input);
const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
const width = view.getUint32(0, true);
const height = view.getUint32(4, true);
const rgba = new Uint8Array(bytes.buffer, bytes.byteOffset + 8, width * height * 4);
const tensor = preprocess(rgba, width, height);
writeFileSync(output, new Uint8Array(tensor.buffer, tensor.byteOffset, tensor.byteLength));
