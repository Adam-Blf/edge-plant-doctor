import labelsJson from "../assets/model/labels.json";

export interface Label {
  id: string;
  crop: string;
  condition: string;
  healthy: boolean;
}

export const LABELS: Label[] = labelsJson;

/** Below this top probability the answer is flagged as uncertain, as in the PWA. */
export const UNCERTAIN_BELOW = 0.6;

export const title = (label: Label) => `${label.crop}, ${label.condition.toLowerCase()}`;

/** French percentage with a comma and a non-breaking space, one decimal under 10 %. */
export const percent = (p: number) =>
  `${(p * 100).toFixed(p < 0.1 ? 1 : 0).replace(".", ",")}\u00a0%`;
