import { AlertTriangle, Camera, CheckCircle, Cpu, Gallery, Leaf, WifiOff } from "reicon";

type IconFactory = (options?: { size?: number; weight?: "Outline" | "Filled" }) => SVGSVGElement;

const ICONS = {
  alert: AlertTriangle,
  camera: Camera,
  check: CheckCircle,
  cpu: Cpu,
  gallery: Gallery,
  leaf: Leaf,
  offline: WifiOff,
} satisfies Record<string, IconFactory>;

/** Reicon, Outline weight everywhere. Icons beside a label stay decorative. */
export function icon(name: keyof typeof ICONS, size = 24): SVGSVGElement {
  const svg = ICONS[name]({ size });
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  return svg;
}
