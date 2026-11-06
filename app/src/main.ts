import "./style.css";
import { Classifier } from "./classifier";
import { loadImage } from "./image";
import { icon } from "./icons";
import { renderResult, showError, type Label } from "./ui";

declare const __APP_VERSION__: string;

const BASE = import.meta.env.BASE_URL;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

function setButton(button: HTMLElement, name: "camera" | "gallery", text: string) {
  button.replaceChildren(icon(name), document.createTextNode(text));
}

function setStatus(state: "loading" | "ready" | "failed", text: string) {
  const chip = el("status");
  chip.dataset.state = state;
  chip.replaceChildren(...(state === "ready" ? [icon("offline", 16)] : []), document.createTextNode(text));
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  navigator.serviceWorker.register(`${BASE}sw.js`, { scope: BASE }).catch(() => {
    setStatus("ready", "Prêt, hors ligne non disponible");
  });
}

function showIosHint() {
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
  const standalone = "standalone" in navigator && (navigator as { standalone?: boolean }).standalone;
  el("ios-hint").hidden = !ios || Boolean(standalone);
}

async function main() {
  el("brand").prepend(icon("leaf", 26));
  setButton(el("btn-camera"), "camera", "Diagnostiquer ma feuille en photo");
  setButton(el("btn-gallery"), "gallery", "Diagnostiquer une photo de ma galerie");
  showIosHint();
  registerServiceWorker();

  let labels: Label[];
  let classifier: Classifier;
  try {
    [labels, classifier] = await Promise.all([
      fetch(`${BASE}model/labels.json`).then((r) => r.json() as Promise<Label[]>),
      Classifier.load(BASE),
    ]);
  } catch {
    setStatus("failed", "Modèle indisponible");
    showError("Le modèle n'a pas pu être chargé. Rechargez la page avec une connexion.");
    return;
  }
  const size = await fetch(`${BASE}model/plant-doctor-int8.onnx`, { method: "HEAD" })
    .then((r) => Number(r.headers.get("content-length") ?? 0))
    .catch(() => 0);
  el("model-info").textContent =
    `Version ${__APP_VERSION__}. MobileNetV3-Small quantifié en int8` +
    (size ? `, ${(size / 1e6).toFixed(1).replace(".", ",")} Mo.` : ".");
  setStatus("ready", "Prêt, fonctionne hors ligne");

  const buttons = [el<HTMLButtonElement>("btn-camera"), el<HTMLButtonElement>("btn-gallery")];
  for (const b of buttons) b.disabled = false;

  async function analyse(file: File | undefined) {
    if (!file) return;
    for (const b of buttons) b.disabled = true;
    try {
      const { tensor, preview } = await loadImage(file);
      const outcome = await classifier.classify(tensor);
      renderResult(labels, outcome, preview);
    } catch {
      showError("Cette image n'a pas pu être lue. Essayez une autre photo.");
    } finally {
      for (const b of buttons) b.disabled = false;
    }
  }

  for (const [button, input] of [
    ["btn-camera", "input-camera"],
    ["btn-gallery", "input-gallery"],
  ] as const) {
    const field = el<HTMLInputElement>(input);
    el(button).addEventListener("click", () => field.click());
    field.addEventListener("change", () => {
      void analyse(field.files?.[0]);
      field.value = "";
    });
  }
}

void main();
