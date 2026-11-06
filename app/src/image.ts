import { preprocess } from "./preprocess";

/** Longest side handed to the pixel pipeline: four times the model input, an exact 4:1 box average. */
const MAX_SIDE = 896;

export interface LoadedImage {
  tensor: Float32Array;
  /** Square crop at preview resolution, ready to draw. */
  preview: HTMLCanvasElement;
}

/**
 * Decode a photo, take its centre square, and run the shared preprocessing.
 *
 * The browser draws the crop at no more than 896 px (a 48 MP iPhone photo would exceed
 * Safari's canvas limit otherwise), then `preprocess` does the exact area resize.
 * An <img> element is used rather than createImageBitmap so EXIF orientation is applied
 * the same way on every Safari version.
 */
export async function loadImage(file: File): Promise<LoadedImage> {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const side = Math.min(img.naturalWidth, img.naturalHeight);
    const target = Math.min(side, MAX_SIDE);
    const canvas = document.createElement("canvas");
    canvas.width = target;
    canvas.height = target;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Canvas indisponible");
    const sx = (img.naturalWidth - side) / 2;
    const sy = (img.naturalHeight - side) / 2;
    ctx.drawImage(img, sx, sy, side, side, 0, 0, target, target);
    const { data } = ctx.getImageData(0, 0, target, target);
    return { tensor: preprocess(data, target, target), preview: canvas };
  } finally {
    URL.revokeObjectURL(url);
  }
}
