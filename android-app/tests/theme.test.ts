import { DARK, LIGHT, type Theme } from "../src/theme";

function luminance(hex: string): number {
  const channel = (i: number) => {
    const c = parseInt(hex.slice(1 + i * 2, 3 + i * 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(0) + 0.7152 * channel(1) + 0.0722 * channel(2);
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** [foreground, background] pairs that carry text. */
const TEXT_PAIRS: [keyof Theme, keyof Theme][] = [
  ["ink", "bg"],
  ["ink", "card"],
  ["muted", "bg"],
  ["muted", "card"],
  ["leaf", "bg"],
  ["leaf", "card"],
  ["ok", "card"],
  ["sick", "bg"],
  ["sick", "card"],
  ["onLeaf", "leaf"],
];

describe.each([
  ["light", LIGHT],
  ["dark", DARK],
])("%s theme contrast", (_name, theme) => {
  it.each(TEXT_PAIRS)("%s on %s reaches 4.5:1", (fg, bg) => {
    expect(contrast(theme[fg], theme[bg])).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps the progress track visible against the card (3:1)", () => {
    expect(contrast(theme.leaf, theme.track)).toBeGreaterThanOrEqual(3);
  });
});
