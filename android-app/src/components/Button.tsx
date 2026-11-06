import { Pressable, StyleSheet, Text } from "react-native";
import { Icon } from "../icons/Icon";
import type { IconName } from "../icons/paths";
import { FONTS, useTheme } from "../theme";

interface Props {
  label: string;
  icon: IconName;
  variant: "primary" | "secondary";
  disabled?: boolean;
  onPress: () => void;
}

export function Button({ label, icon, variant, disabled, onPress }: Props) {
  const t = useTheme();
  const primary = variant === "primary";
  const fg = primary ? t.onLeaf : t.leaf;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled: Boolean(disabled) }}
      disabled={disabled}
      onPress={onPress}
      android_ripple={{ color: primary ? t.onLeaf + "33" : t.leaf + "33" }}
      style={({ pressed }) => [
        styles.btn,
        { borderColor: t.leaf, backgroundColor: primary ? t.leaf : "transparent" },
        pressed && styles.pressed,
        disabled && styles.disabled,
      ]}
    >
      <Icon name={icon} color={fg} size={24} />
      <Text style={[styles.label, { color: fg }]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  btn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    minHeight: 52,
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 16,
    borderWidth: 2,
    overflow: "hidden",
  },
  label: { fontFamily: FONTS.bold, fontSize: 16, flexShrink: 1, textAlign: "center" },
  pressed: { transform: [{ scale: 0.98 }] },
  disabled: { opacity: 0.5 },
});
