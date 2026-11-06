import { decode } from "jpeg-js";

export interface Rgba {
  data: Uint8Array;
  width: number;
  height: number;
}

/** Base64 string to bytes. `atob` is global on Hermes and on Node 16+. */
export function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/** Decode a base64 JPEG into RGBA bytes. */
export function decodeJpegBase64(base64: string): Rgba {
  const { data, width, height } = decode(base64ToBytes(base64), {
    useTArray: true,
    formatAsRGBA: true,
    maxMemoryUsageInMB: 512,
  });
  return { data, width, height };
}
