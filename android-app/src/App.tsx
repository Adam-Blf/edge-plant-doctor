import { useFonts } from "expo-font";
import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import { useCallback, useEffect, useRef, useState } from "react";
import { ScrollView, StyleSheet, Text, View, type ScrollViewInstance } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Button } from "./components/Button";
import { ResultCard } from "./components/ResultCard";
import { StatusChip, type ModelState } from "./components/StatusChip";
import { Icon } from "./icons/Icon";
import { Classifier, type Classification } from "./inference";
import { prepareImage } from "./image";
import { DARK, FONTS, useTheme } from "./theme";

interface Outcome {
  previewUri: string;
  classification: Classification;
  preprocessMs: number;
}

const PICKER_OPTIONS: ImagePicker.ImagePickerOptions = {
  mediaTypes: ["images"],
  allowsEditing: false,
  quality: 1,
};

export default function App() {
  const t = useTheme();
  const [fontsReady] = useFonts({
    [FONTS.medium]: require("../assets/fonts/BeVietnamPro-500.ttf"),
    [FONTS.bold]: require("../assets/fonts/BeVietnamPro-700.ttf"),
    [FONTS.extraBold]: require("../assets/fonts/BeVietnamPro-800.ttf"),
  });
  const [state, setState] = useState<ModelState>("loading");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const classifier = useRef<Classifier | null>(null);
  const scroll = useRef<ScrollViewInstance>(null);
  const resultTop = useRef(0);

  useEffect(() => {
    Classifier.load()
      .then((c) => {
        classifier.current = c;
        setState("ready");
      })
      .catch(() => {
        setState("failed");
        setError("Le modèle n'a pas pu être chargé. Relancez l'application.");
      });
  }, []);

  const analyse = useCallback(async (source: "camera" | "gallery") => {
    const model = classifier.current;
    if (!model) return;
    setError(null);
    setBusy(true);
    try {
      if (source === "camera") {
        const permission = await ImagePicker.requestCameraPermissionsAsync();
        if (!permission.granted) {
          setError("L'accès à l'appareil photo est refusé. Autorisez-le dans les réglages d'Android, ou choisissez une photo de votre galerie.");
          return;
        }
      }
      const picked =
        source === "camera"
          ? await ImagePicker.launchCameraAsync(PICKER_OPTIONS)
          : await ImagePicker.launchImageLibraryAsync(PICKER_OPTIONS);
      if (picked.canceled || !picked.assets[0]) return;

      const started = performance.now();
      const { tensor, previewUri } = await prepareImage(picked.assets[0].uri);
      const preprocessMs = performance.now() - started;
      const classification = await model.classify(tensor);
      setOutcome({ previewUri, classification, preprocessMs });
      requestAnimationFrame(() => scroll.current?.scrollTo({ y: resultTop.current, animated: true }));
    } catch {
      setError("Cette image n'a pas pu être lue. Essayez une autre photo.");
    } finally {
      setBusy(false);
    }
  }, []);

  if (!fontsReady) return <View style={[styles.fill, { backgroundColor: t.bg }]} />;

  const disabled = state !== "ready" || busy;
  return (
    <SafeAreaView style={[styles.fill, { backgroundColor: t.bg }]}>
      <StatusBar style={t === DARK ? "light" : "dark"} />
      <ScrollView ref={scroll} contentContainerStyle={styles.content}>
        <View style={styles.top}>
          <View style={styles.brand}>
            <Icon name="leaf" color={t.leaf} size={26} />
            <Text style={[styles.brandName, { color: t.ink }]}>Plant Doctor</Text>
          </View>
          <StatusChip state={state} />
        </View>

        <Text accessibilityRole="header" style={[styles.h1, { color: t.ink }]}>
          Votre plante est-elle malade ?
        </Text>
        <Text style={[styles.lead, { color: t.muted }]}>
          Photographiez une feuille. La réponse arrive en moins d'une seconde, directement sur ce
          téléphone, sans réseau.
        </Text>

        <View style={styles.actions}>
          <Button
            variant="primary"
            icon="camera"
            label={busy ? "Analyse de votre feuille" : "Diagnostiquer ma feuille en photo"}
            disabled={disabled}
            onPress={() => void analyse("camera")}
          />
          <Button
            variant="secondary"
            icon="gallery"
            label="Diagnostiquer une photo de ma galerie"
            disabled={disabled}
            onPress={() => void analyse("gallery")}
          />
        </View>

        {error && (
          <Text accessibilityRole="alert" style={[styles.error, { color: t.sick, borderColor: t.sick }]}>
            {error}
          </Text>
        )}

        {outcome && (
          <View onLayout={(e) => (resultTop.current = e.nativeEvent.layout.y)}>
            <ResultCard
              previewUri={outcome.previewUri}
              outcome={outcome.classification}
              preprocessMs={outcome.preprocessMs}
            />
          </View>
        )}

        <View style={styles.foot}>
          <Text style={[styles.footText, { color: t.muted }]}>
            MobileNetV3-Small quantifié en int8, exécuté par ONNX Runtime natif.
          </Text>
          <Text style={[styles.footText, { color: t.muted }]}>
            Entraîné sur PlantVillage, des photos de feuilles prises en laboratoire. Sur le terrain,
            avec un fond chargé ou plusieurs feuilles, la réponse peut se tromper. Aucune image ne
            quitte votre téléphone.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  content: { maxWidth: 544, width: "100%", alignSelf: "center", padding: 20, paddingBottom: 32 },
  top: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" },
  brand: { flexDirection: "row", alignItems: "center", gap: 8 },
  brandName: { fontFamily: FONTS.extraBold, fontSize: 18, letterSpacing: -0.2 },
  h1: { fontFamily: FONTS.extraBold, fontSize: 36, lineHeight: 38, letterSpacing: -1, marginTop: 40, marginBottom: 12 },
  lead: { fontFamily: FONTS.medium, fontSize: 17, lineHeight: 25, marginBottom: 24 },
  actions: { gap: 12 },
  error: { fontFamily: FONTS.medium, fontSize: 15, marginTop: 24, padding: 12, borderWidth: 1.5, borderRadius: 16 },
  foot: { marginTop: 40, gap: 8 },
  footText: { fontFamily: FONTS.medium, fontSize: 13, lineHeight: 19 },
});
