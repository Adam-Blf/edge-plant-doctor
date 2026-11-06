/**
 * Image preprocessing, a port of app/src/preprocess.ts of the PWA (itself a line-for-line mirror
 * of src/edgeplant/preprocess.py): centre crop to a square, area-weighted resize to 224x224,
 * ImageNet normalisation. The numerical code is kept identical on purpose so both apps share
 * one contract. On Android the phone first hands over a 896 px square (see image.ts).
 */

export const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406].map(Math.fround);
const STD = [0.229, 0.224, 0.225].map(Math.fround);

type Pixels = Uint8ClampedArray | Uint8Array;

interface Span {
  first: number;
  weights: Float64Array;
}

/** For each output pixel, the source span it covers and its normalised overlap weights. */
export function areaSpans(src: number, dst: number): Span[] {
  const scale = src / dst;
  return Array.from({ length: dst }, (_, i) => {
    const lo = i * scale;
    const hi = (i + 1) * scale;
    const first = Math.floor(lo);
    const last = Math.min(Math.ceil(hi), src);
    const weights = new Float64Array(last - first);
    let total = 0;
    for (let j = first; j < last; j++) {
      const w = Math.min(hi, j + 1) - Math.max(lo, j);
      weights[j - first] = w;
      total += w;
    }
    for (let k = 0; k < weights.length; k++) weights[k] /= total;
    return { first, weights };
  });
}

/** Square RGBA (side x side) to RGB float (size x size x 3, range 0..255). */
export function resizeArea(rgba: Pixels, side: number, size = SIZE): Float32Array {
  const spans = areaSpans(side, size);
  const rows = new Float64Array(size * side * 3);
  for (let i = 0; i < size; i++) {
    const { first, weights } = spans[i];
    for (let x = 0; x < side; x++) {
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let k = 0; k < weights.length; k++) {
          acc += weights[k] * rgba[((first + k) * side + x) * 4 + c];
        }
        rows[(i * side + x) * 3 + c] = acc;
      }
    }
  }
  const out = new Float32Array(size * size * 3);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const { first, weights } = spans[j];
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let k = 0; k < weights.length; k++) {
          acc += weights[k] * rows[(i * side + first + k) * 3 + c];
        }
        out[(i * size + j) * 3 + c] = acc;
      }
    }
  }
  return out;
}

/** Crop the largest centred square out of an RGBA image. */
export function centerCropSquare(rgba: Pixels, width: number, height: number) {
  const side = Math.min(width, height);
  if (width === height) return { data: rgba, side };
  const top = Math.floor((height - side) / 2);
  const left = Math.floor((width - side) / 2);
  const data = new Uint8Array(side * side * 4);
  for (let y = 0; y < side; y++) {
    const from = ((top + y) * width + left) * 4;
    data.set(rgba.subarray(from, from + side * 4), y * side * 4);
  }
  return { data, side };
}

/** 0..255 HWC floats to a normalised NCHW tensor with a batch of one. */
export function normalize(hwc: Float32Array, size = SIZE): Float32Array {
  const out = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let p = 0; p < plane; p++) {
    for (let c = 0; c < 3; c++) {
      const unit = Math.fround(hwc[p * 3 + c] / 255);
      out[c * plane + p] = Math.fround(Math.fround(unit - MEAN[c]) / STD[c]);
    }
  }
  return out;
}

/** Full pipeline: RGBA pixels of any size to the (1, 3, 224, 224) model input. */
export function preprocess(rgba: Pixels, width: number, height: number): Float32Array {
  const { data, side } = centerCropSquare(rgba, width, height);
  return normalize(resizeArea(data, side));
}
