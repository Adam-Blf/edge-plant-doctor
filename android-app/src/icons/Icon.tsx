import Svg, { Path } from "react-native-svg";
import { ICON_PATHS, type IconName } from "./paths";

interface Props {
  name: IconName;
  color: string;
  size?: number;
}

/** Reicon Outline glyph. Always decorative: the label next to it carries the meaning. */
export function Icon({ name, color, size = 24 }: Props) {
  return (
    <Svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      accessibilityElementsHidden
      importantForAccessibility="no-hide-descendants"
    >
      {ICON_PATHS[name].map((p, i) => (
        <Path
          key={i}
          d={p.d}
          fill={color}
          fillRule={p.evenodd ? "evenodd" : "nonzero"}
          clipRule={p.evenodd ? "evenodd" : "nonzero"}
        />
      ))}
    </Svg>
  );
}
