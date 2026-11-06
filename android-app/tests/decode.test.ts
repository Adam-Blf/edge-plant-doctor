import { encode } from "jpeg-js";
import { decodeJpegBase64, base64ToBytes } from "../src/decode";
import { preprocess } from "../src/preprocess";

function jpegBase64(width: number, height: number, rgb: [number, number, number]): string {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = rgb[0];
    data[i * 4 + 1] = rgb[1];
    data[i * 4 + 2] = rgb[2];
    data[i * 4 + 3] = 255;
  }
  const bytes = encode({ data, width, height }, 100).data;
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary);
}

describe("JPEG decoding", () => {
  it("round-trips base64 bytes", () => {
    expect(Array.from(base64ToBytes("AQID"))).toEqual([1, 2, 3]);
  });

  it("decodes a base64 JPEG to RGBA of the right size", () => {
    const { data, width, height } = decodeJpegBase64(jpegBase64(16, 16, [200, 100, 50]));
    expect([width, height]).toEqual([16, 16]);
    expect(data.length).toBe(16 * 16 * 4);
    expect(Math.abs(data[0] - 200)).toBeLessThanOrEqual(3);
    expect(data[3]).toBe(255);
  });

  it("feeds the preprocessing to a 1x3x224x224 tensor", () => {
    const { data, width, height } = decodeJpegBase64(jpegBase64(32, 32, [124, 116, 104]));
    const tensor = preprocess(data, width, height);
    expect(tensor.length).toBe(3 * 224 * 224);
    expect(Math.abs(tensor[0])).toBeLessThan(0.1);
  });
});
