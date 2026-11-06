"""Markdown tables and matplotlib figures built only from the JSON files in results/."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import RESULTS_DIR, ROOT  # noqa: E402

INK, LEAF, RUST, SAND, GREY = "#12261a", "#1f6b3a", "#a53a1e", "#d9a441", "#8a948c"
VARIANTS = {
    "": "FP32",
    "dynamic_gemm": "Dynamique, classifieur seul",
    "dynamic_all": "Dynamique, tout (naïf)",
    "dynamic_mixed": "Dynamique, mixte",
    "static_all_minmax": "Statique, tout (naïf)",
    "static_mixed_minmax": "Statique mixte, min-max",
    "static_mixed_percentile": "Statique mixte, percentile",
    "static_mixed_entropy": "Statique mixte, entropie",
}
STUDENTS = {"student_baseline": "Élève seul", "student_distilled": "Élève distillé"}


def load(name: str) -> dict:
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def label(stem: str) -> str:
    base, _, variant = stem.partition(".")
    return f"{STUDENTS.get(base, base)}, {VARIANTS.get(variant, variant)}"


def accuracy_table(acc: dict, latency: dict) -> str:
    one = latency.get("threads", {}).get("1", {})
    rows = [
        "| Modèle | Précision | F1 macro | Taille (Mo) | p50 x86 (ms) | p95 x86 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stem, r in acc.items():
        lat = one.get(stem, {})
        p50 = f"{lat['p50_ms']:.1f}" if lat else "-"
        p95 = f"{lat['p95_ms']:.1f}" if lat else "-"
        rows.append(
            f"| {label(stem)} | {r['accuracy'] * 100:.2f} % | {r['macro_f1'] * 100:.2f} % "
            f"| {r['size_mb']:.2f} | {p50} | {p95} |"
        )
    return "\n".join(rows)


FAMILY_FR = {
    "stem": "Convolution d'entrée",
    "depthwise": "Convolutions en profondeur",
    "pointwise": "Convolutions ponctuelles",
    "classifier": "Classifieur",
}


def ablation_table(ablation: dict) -> str:
    if not ablation:
        return "Étude par famille de couches : non mesurée."
    lines = [
        f"Une seule famille en int8, le reste en flottant ({ablation['images']} images de "
        f"validation, FP32 : {ablation['fp32_accuracy']:.2%})",
        "",
        "| Famille quantifiée | Couches | Précision | Accord avec FP32 |",
        "|---|---:|---:|---:|",
    ]
    for family, row in ablation["families"].items():
        lines.append(
            f"| {FAMILY_FR.get(family, family)} | {row['nodes']} | {row['accuracy']:.2%} "
            f"| {row['agreement_with_fp32']:.2%} |"
        )
    return "\n".join(lines).replace("%", " %")


def latency_table(latency: dict, title: str) -> str:
    if not latency:
        return f"{title} : non mesuré."
    host = latency["host"]
    lines = [
        f"{title} ({host['machine']}, onnxruntime {host['onnxruntime']}"
        + (", émulé" if host["emulated"] else "")
        + ")",
        "",
        "| Modèle | Threads | p50 (ms) | p95 (ms) |",
        "|---|---:|---:|---:|",
    ]
    for threads, models in latency["threads"].items():
        for stem, r in models.items():
            lines.append(f"| {label(stem)} | {threads} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |")
    return "\n".join(lines)


def plot_size_accuracy(acc: dict, out: Path) -> None:
    # Left: full scale, where the naive recipes collapse. Right: zoom on the usable
    # variants, labelled once (distilled) with staggered offsets so labels never overlap.
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(9, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
    usable = {s: r for s, r in acc.items() if r["accuracy"] >= 0.9}
    for ax, rows in ((full, acc), (zoom, usable)):
        for stem, r in rows.items():
            distilled = stem.startswith("student_distilled")
            ax.scatter(
                r["size_mb"],
                r["accuracy"] * 100,
                s=60,
                color=LEAF if distilled else RUST,
                marker="o" if "." not in stem else "s",
                alpha=0.9,
            )
        ax.set_xlabel("Taille du fichier (Mo)")
        ax.grid(alpha=0.25)
    naive: dict[str, tuple[float, float]] = {}
    for stem, r in acc.items():
        if r["accuracy"] < 0.9:
            variant = stem.partition(".")[2]
            point = (r["size_mb"], r["accuracy"] * 100)
            naive[variant] = max(naive.get(variant, point), point, key=lambda p: p[1])
    for variant, point in naive.items():
        full.annotate(
            VARIANTS.get(variant, variant),
            point,
            textcoords="offset points",
            xytext=(8, 2),
            fontsize=7,
            color=INK,
        )
    callouts = {
        "student_distilled.static_mixed_percentile": ("Modèle livré (int8)", (40, -30)),
        "student_distilled": ("Distillé FP32", (-70, -30)),
    }
    for stem, (text, offset) in callouts.items():
        if stem in usable:
            r = usable[stem]
            zoom.annotate(
                text,
                (r["size_mb"], r["accuracy"] * 100),
                textcoords="offset points",
                xytext=offset,
                fontsize=8,
                color=INK,
                fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.6},
            )
    full.set_ylabel("Précision top-1 sur le jeu de test (%)")
    full.set_title("Toutes les variantes", fontsize=9)
    zoom.set_title("Variantes utilisables (zoom)", fontsize=9)
    fig.suptitle("Précision et taille : vert = distillé, rouge = élève seul", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_latency(latency: dict, emulated: dict, out: Path) -> None:
    # x86 in milliseconds and QEMU in seconds never share an axis: left panel shows x86
    # latency per variant, right panel the arm64 speed-up relative to each student's FP32.
    one = latency["threads"]["1"]
    models = sorted(one, key=lambda m: one[m]["p50_ms"])
    em = emulated.get("threads", {}).get("1", {}) if emulated else {}
    fig, (left, right) = (
        plt.subplots(1, 2, figsize=(9, 4.8), gridspec_kw={"width_ratios": [1.6, 1]} if em else None)
        if em
        else (*plt.subplots(figsize=(7.5, 4.8)), None)
    )
    left.barh(
        [label(m) for m in models],
        [one[m]["p50_ms"] for m in models],
        color=[LEAF if m.startswith("student_distilled") else RUST for m in models],
    )
    left.invert_yaxis()
    left.tick_params(axis="y", labelsize=6.5)
    left.set_xlabel("Latence p50 x86, 1 thread (ms)")
    left.grid(axis="x", alpha=0.25)
    left.set_title("x86 : vert = distillé, rouge = élève seul", fontsize=9)
    if right is not None:
        names = [m for m in em if not m.endswith(("student_baseline", "student_distilled"))]
        base = {m: em.get(m.partition(".")[0], {}).get("p50_ms") for m in names}
        speed = [base[m] / em[m]["p50_ms"] if base[m] else 0 for m in names]
        right.bar([VARIANTS.get(m.partition(".")[2], m) for m in names], speed, color=SAND)
        right.axhline(1, color=INK, lw=0.8, ls="--")
        right.set_ylabel("Accélération vs FP32 (x)")
        right.set_title("arm64 émulé QEMU (rapport seul)", fontsize=9)
        right.tick_params(axis="x", labelsize=7, rotation=20)
        right.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_training(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for tag, color in (("student_baseline", RUST), ("student_distilled", LEAF), ("teacher", GREY)):
        hist = load(f"train_{tag}.json").get("history", [])
        if hist:
            ax.plot(
                [h["epoch"] for h in hist],
                [h["accuracy"] * 100 for h in hist],
                marker="o",
                color=color,
                label={**STUDENTS, "teacher": "Professeur EfficientNet-B0"}.get(tag, tag),
            )
    ax.set_xlabel("Époque")
    ax.set_ylabel("Précision de validation (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build docs/results.md tables and figures")
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    args = ap.parse_args()
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    acc, lat, arm = (
        load("accuracy.json"),
        load("latency_x86.json"),
        load("latency_arm64_emulated.json"),
    )
    plot_size_accuracy(acc, figures / "size_accuracy.png")
    plot_training(figures / "training.png")
    if lat:
        plot_latency(lat, arm, figures / "latency.png")
    tables = [
        accuracy_table(acc, lat),
        latency_table(lat, "Latence x86"),
        latency_table(arm, "Latence arm64 émulée"),
        ablation_table(load("ablation_families.json")),
    ]
    (args.out / "results_tables.md").write_text("\n\n".join(tables) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
