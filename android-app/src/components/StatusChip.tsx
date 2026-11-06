import { StyleSheet, Text, View } from "react-native";
import { Icon } from "../icons/Icon";
import { FONTS, useTheme } from "../theme";

export type ModelState = "loading" | "ready" | "failed";

const TEXT: Record<ModelState, string> = {
  loading: "Chargement du modèle",
  ready: "Prêt, fonctionne hors ligne",
  failed: "Modèle indisponible",
};

export function StatusChip({ state }: { state: ModelState }) {
  const t = useTheme();
  const color = state === "ready" ? t.ok : state === "failed" ? t.sick : t.muted;
  return (
    <View
      accessibilityRole="text"
      accessibilityLiveRegion="polite"
      style={[styles.chip, { borderColor: state === "loading" ? t.line : color }]}
    >
      {state === "ready" && <Icon name="offline" color={color} size={16} />}
      <Text style={[styles.text, { color }]}>{TEXT[state]}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderRadius: 999,
  },
  text: { fontFamily: FONTS.medium, fontSize: 13 },
});
