import type { Classification } from "./classifier";
import { icon } from "./icons";

export interface Label {
  id: string;
  crop: string;
  condition: string;
  healthy: boolean;
}

const UNCERTAIN_BELOW = 0.6;

function node<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function byId<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

const percent = (p: number) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0).replace(".", ",")} %`;
const title = (l: Label) => `${l.crop}, ${l.condition.toLowerCase()}`;

export function showError(message: string) {
  const box = byId("error");
  box.textContent = message;
  box.hidden = false;
}

/** Fills the result card. Every string goes through textContent, never innerHTML. */
export function renderResult(labels: Label[], outcome: Classification, preview: HTMLCanvasElement) {
  byId("error").hidden = true;
  const best = outcome.top[0];
  const label = labels[best.index];

  const target = byId<HTMLCanvasElement>("preview");
  target.width = 448;
  target.height = 448;
  target.getContext("2d")?.drawImage(preview, 0, 0, 448, 448);

  const pill = byId("pill");
  pill.dataset.kind = label.healthy ? "healthy" : "sick";
  pill.replaceChildren(
    icon(label.healthy ? "check" : "alert", 16),
    document.createTextNode(label.healthy ? "Feuille saine" : "Maladie probable"),
  );
  byId("headline").textContent = title(label);

  const uncertain = byId("uncertain");
  uncertain.hidden = best.probability >= UNCERTAIN_BELOW;
  uncertain.textContent = "Résultat incertain. Cadrez une seule feuille, bien éclairée, sur un fond simple.";

  const bars = byId("bars");
  bars.replaceChildren(
    ...outcome.top.map((p) => {
      const row = node("li", "bar-row");
      row.append(node("span", "name", title(labels[p.index])), node("span", "value", percent(p.probability)));
      const track = node("div", "bar-track");
      const fill = node("div", "bar-fill");
      track.append(fill);
      row.append(track);
      requestAnimationFrame(() => {
        fill.style.width = `${(p.probability * 100).toFixed(1)}%`;
      });
      return row;
    }),
  );
  byId("timing").textContent = `Analyse en ${Math.round(outcome.milliseconds)} ms sur cet appareil.`;
  byId("result").hidden = false;
  byId("result").scrollIntoView({ behavior: "smooth", block: "start" });
}
