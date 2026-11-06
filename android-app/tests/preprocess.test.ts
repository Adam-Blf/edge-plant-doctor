import { areaSpans, centerCropSquare, normalize, preprocess, resizeArea, SIZE } from "../src/preprocess";

/** Deterministic RGBA test pattern, identical to the one used to produce the golden values. */
function synth(w: number, h: number) {
  const px = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let c = 0; c < 4; c++) px[(y * w + x) * 4 + c] = c === 3 ? 255 : (x * 37 + y * 17 + c * 91) % 256;
  return px;
}

function solid(side: number, value: number) {
  return new Uint8Array(side * side * 4).fill(value);
}

/**
 * Golden values: the output of app/src/preprocess.ts of the PWA (edge-plant-doctor-pwa,
 * run under Node) on synth(w, h). The Android port must reproduce them to float32 precision.
 */
const GOLDEN = [
  {
    w: 20,
    h: 14,
    sum: 31060.98654,
    samples: [
      [0, -0.2170563042163849],
      [223, -0.7479236125946045],
      [12345, 1.2899221181869507],
      [50175, -1.3472901582717896],
      [50177, 1.5007002353668213],
      [60000, 0.9054622650146484],
      [100352, -1.1595641374588013],
      [150527, 2.1519827842712402],
    ],
  },
  {
    w: 14,
    h: 14,
    sum: 31134.24868,
    samples: [
      [0, -2.1179039478302],
      [223, 1.7351655960083008],
      [12345, -0.6109256148338318],
      [50176, -0.44257697463035583],
      [100351, -1.598039150238037],
      [112000, 2.2565577030181885],
      [150527, 0.21734212338924408],
    ],
  },
  {
    w: 896,
    h: 896,
    sum: 34104.51778,
    samples: [
      [0, -0.7307989001274109],
      [1, 0.15968838334083557],
      [224, 0.43368443846702576],
      [12345, -0.4568028450012207],
      [50176, 0.9754902720451355],
      [100351, 0.5553222298622131],
      [112000, 0.8273640275001526],
      [150527, -0.42753803730010986],
    ],
  },
];

describe("parity with the PWA preprocessing", () => {
  it.each(GOLDEN)("matches the PWA tensor on a $w x $h image", ({ w, h, sum, samples }) => {
    const tensor = preprocess(synth(w, h), w, h);
    expect(tensor).toBeInstanceOf(Float32Array);
    expect(tensor.length).toBe(3 * SIZE * SIZE);
    for (const [index, expected] of samples) expect(tensor[index]).toBeCloseTo(expected, 6);
    expect(tensor.reduce((a, b) => a + b, 0)).toBeCloseTo(sum, 3);
  });
});

describe("area resize", () => {
  it("rows of weights sum to one", () => {
    for (const [src, dst] of [
      [256, 224],
      [896, 224],
      [100, 224],
    ]) {
      for (const span of areaSpans(src, dst)) {
        expect(span.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      }
    }
  });

  it("uses exactly four source pixels per output pixel from 896 to 224", () => {
    for (const span of areaSpans(896, 224)) {
      expect(span.weights.length).toBe(4);
      for (const w of span.weights) expect(w).toBeCloseTo(0.25, 12);
    }
  });

  it("returns the exact 4x4 mean of the source area on a 896 to 224 reduction", () => {
    const side = 896;
    const img = synth(side, side);
    const out = resizeArea(img, side);
    for (const [oy, ox] of [
      [0, 0],
      [0, 223],
      [100, 57],
      [223, 223],
    ]) {
      for (let c = 0; c < 3; c++) {
        let total = 0;
        for (let dy = 0; dy < 4; dy++)
          for (let dx = 0; dx < 4; dx++) total += img[((oy * 4 + dy) * side + ox * 4 + dx) * 4 + c];
        expect(out[(oy * SIZE + ox) * 3 + c]).toBeCloseTo(total / 16, 3);
      }
    }
  });

  it("keeps a constant image constant", () => {
    const out = resizeArea(solid(300, 77), 300);
    expect(out.length).toBe(SIZE * SIZE * 3);
    for (const v of out) expect(v).toBeCloseTo(77, 3);
  });
});

describe("crop and normalise", () => {
  it("crops a wide image to its centre square", () => {
    const w = 300;
    const h = 100;
    const img = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 100; x < 200; x++) img[(y * w + x) * 4] = 255;
    const { data, side } = centerCropSquare(img, w, h);
    expect(side).toBe(100);
    expect(data.length).toBe(100 * 100 * 4);
    expect(data.filter((_, i) => i % 4 === 0).every((v) => v === 255)).toBe(true);
  });

  it("normalises white and black with the ImageNet statistics", () => {
    const white = normalize(new Float32Array(SIZE * SIZE * 3).fill(255));
    expect(white[0]).toBeCloseTo((1 - 0.485) / 0.229, 5);
    expect(white[SIZE * SIZE]).toBeCloseTo((1 - 0.456) / 0.224, 5);
    expect(white[2 * SIZE * SIZE]).toBeCloseTo((1 - 0.406) / 0.225, 5);
    const black = normalize(new Float32Array(SIZE * SIZE * 3));
    expect(black[0]).toBeCloseTo(-0.485 / 0.229, 5);
    expect(black[2 * SIZE * SIZE]).toBeCloseTo(-0.406 / 0.225, 5);
  });

  it("lays the tensor out channel first (NCHW)", () => {
    const hwc = new Float32Array(SIZE * SIZE * 3);
    hwc[3 * 5 + 1] = 255; // pixel 5, green channel
    const out = normalize(hwc);
    const plane = SIZE * SIZE;
    expect(out[plane + 5]).toBeCloseTo((1 - 0.456) / 0.224, 5);
    expect(out[5]).toBeCloseTo(-0.485 / 0.229, 5);
  });
});
