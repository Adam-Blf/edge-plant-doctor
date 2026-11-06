import { Image, StyleSheet, Text, View } from "react-native";
import { Icon } from "../icons/Icon";
import { LABELS, UNCERTAIN_BELOW, percent, title } from "../labels";
import type { Classification } from "../inference";
import { FONTS, useTheme } from "../theme";

interface Props {
  previewUri: string;
  outcome: Classification;
  preprocessMs: number;
}

export function ResultCard({ previewUri, outcome, preprocessMs }: Props) {
  const t = useTheme();
  const best = outcome.top[0];
  const label = LABELS[best.index];
  const tone = label.healthy ? t.ok : t.sick;

  return (
    <View
      accessibilityLiveRegion="polite"
      style={[styles.card, { backgroundColor: t.card, borderColor: t.line }]}
    >
      <Image
        source={{ uri: previewUri }}
        accessibilityLabel="Feuille analysée"
        style={[styles.preview, { backgroundColor: t.track }]}
      />
      <View style={[styles.pill, { borderColor: tone }]}>
        <Icon name={label.healthy ? "check" : "alert"} color={tone} size={16} />
        <Text style={[styles.pillText, { color: tone }]}>
          {label.healthy ? "Feuille saine" : "Maladie probable"}
        </Text>
      </View>
      <Text accessibilityRole="header" style={[styles.headline, { color: t.ink }]}>
        {title(label)}
      </Text>
      {best.probability < UNCERTAIN_BELOW && (
        <Text style={[styles.uncertain, { color: t.sick }]}>
          Résultat incertain. Cadrez une seule feuille, bien éclairée, sur un fond simple.
        </Text>
      )}

      <View style={styles.bars}>
        {outcome.top.map((p) => {
          const l = LABELS[p.index];
          return (
            <View
              key={p.index}
              accessible
              accessibilityLabel={`${title(l)}, ${percent(p.probability)}`}
              style={styles.row}
            >
              <View style={styles.rowText}>
                <Text style={[styles.name, { color: t.ink }]}>{title(l)}</Text>
                <Text style={[styles.value, { color: l.healthy ? t.ok : t.sick }]}>
                  {percent(p.probability)}
                </Text>
              </View>
              <View style={[styles.track, { backgroundColor: t.track }]}>
                <View
                  style={[
                    styles.fill,
                    { width: `${Math.max(p.probability * 100, 1)}%`, backgroundColor: l.healthy ? t.ok : t.sick },
                  ]}
                />
              </View>
            </View>
          );
        })}
      </View>

      <Text style={[styles.timing, { color: t.muted }]}>
        {`Analyse en ${Math.round(outcome.milliseconds)} ms sur cet appareil, prétraitement en ${Math.round(preprocessMs)} ms.`}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { gap: 12, marginTop: 32, padding: 16, borderWidth: 1, borderRadius: 24 },
  preview: { width: "100%", aspectRatio: 1, borderRadius: 16 },
  pill: {
    alignSelf: "flex-start",
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 999,
    borderWidth: 1.5,
  },
  pillText: { fontFamily: FONTS.bold, fontSize: 13 },
  headline: { fontFamily: FONTS.extraBold, fontSize: 24, lineHeight: 28, letterSpacing: -0.5 },
  uncertain: { fontFamily: FONTS.medium, fontSize: 15 },
  bars: { gap: 12, marginTop: 4 },
  row: { gap: 4 },
  rowText: { flexDirection: "row", justifyContent: "space-between", gap: 12 },
  name: { fontFamily: FONTS.medium, fontSize: 15, flexShrink: 1 },
  value: { fontFamily: FONTS.bold, fontSize: 15 },
  track: { height: 8, borderRadius: 999, overflow: "hidden" },
  fill: { height: "100%", borderRadius: 999 },
  timing: { fontFamily: FONTS.medium, fontSize: 13, marginTop: 4 },
});
