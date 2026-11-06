import { useColorScheme } from "react-native";

/** Colour tokens of the PWA (app/src/style.css), light and dark. */
export interface Theme {
  bg: string;
  card: string;
  ink: string;
  muted: string;
  line: string;
  leaf: string;
  onLeaf: string;
  sick: string;
  ok: string;
  track: string;
}

export const LIGHT: Theme = {
  bg: "#f3f0e6",
  card: "#fbf9f3",
  ink: "#12261a",
  muted: "#465a4d",
  line: "#d6d1bf",
  leaf: "#1f6b3a",
  onLeaf: "#ffffff",
  sick: "#a53a1e",
  ok: "#1f6b3a",
  track: "#e2ddca",
};

export const DARK: Theme = {
  bg: "#0f1a13",
  card: "#172a1e",
  ink: "#e9efe4",
  muted: "#a9bba9",
  line: "#2c4535",
  leaf: "#6fd08c",
  onLeaf: "#08130c",
  sick: "#ff9a7a",
  ok: "#6fd08c",
  track: "#24392b",
};

export const FONTS = {
  medium: "BeVietnamPro-500",
  bold: "BeVietnamPro-700",
  extraBold: "BeVietnamPro-800",
} as const;

export function useTheme(): Theme {
  return useColorScheme() === "dark" ? DARK : LIGHT;
}
