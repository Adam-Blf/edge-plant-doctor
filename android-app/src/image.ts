import { ImageManipulator, SaveFormat } from "expo-image-manipulator";
import { decodeJpegBase64 } from "./decode";
import { preprocess } from "./preprocess";

/** Side of the intermediate square: exactly four times the 224 px model input, so the final reduction is a 4x4 box average. */
export const INTERMEDIATE_SIDE = 896;

export interface PreparedImage {
  tensor: Float32Array;
  /** file:// URI of the centre square used for the diagnosis, ready to display. */
  previewUri: string;
}

/**
 * Photo URI to model input. React Native has no canvas, so the crop and the first
 * reduction (to 896 px) are done natively by expo-image-manipulator, the pixels are
 * decoded with jpeg-js, and `preprocess` does the exact area average to 224 px.
 * The 896 step is the one deviation from the Python reference, see README.md.
 */
export async function prepareImage(uri: string): Promise<PreparedImage> {
  // First pass only reads the real dimensions, with EXIF orientation applied.
  const full = await ImageManipulator.manipulate(uri).renderAsync();
  const side = Math.min(full.width, full.height);
  const originX = Math.floor((full.width - side) / 2);
  const originY = Math.floor((full.height - side) / 2);

  const square = await ImageManipulator.manipulate(full)
    .crop({ originX, originY, width: side, height: side })
    .resize({ width: INTERMEDIATE_SIDE, height: INTERMEDIATE_SIDE })
    .renderAsync();
  const saved = await square.saveAsync({ format: SaveFormat.JPEG, compress: 1, base64: true });
  if (!saved.base64) throw new Error("Image illisible");

  const { data, width, height } = decodeJpegBase64(saved.base64);
  return { tensor: preprocess(data, width, height), previewUri: saved.uri };
}
