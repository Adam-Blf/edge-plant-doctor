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
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for stem, r in acc.items():
        distilled = stem.startswith("student_distilled")
        ax.scatter(
            r["size_mb"],
            r["accuracy"] * 100,
            s=70,
            color=LEAF if distilled else RUST,
            marker="o" if "." not in stem else "s",
        )
        ax.annotate(
            stem.partition(".")[2] or "fp32",
            (r["size_mb"], r["accuracy"] * 100),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
            color=INK,
        )
    ax.set_xlabel("Taille du fichier (Mo)")
    ax.set_ylabel("Précision top-1 sur le jeu de test (%)")
    ax.set_title("Précision et taille : vert = distillé, rouge = élève seul", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_latency(latency: dict, emulated: dict, out: Path) -> None:
    models = list(latency["threads"]["1"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    width = 0.38
    xs = range(len(models))
    ax.bar(
        [x - width / 2 for x in xs],
        [latency["threads"]["1"][m]["p50_ms"] for m in models],
        width,
        color=LEAF,
        label="x86, 1 thread (p50)",
    )
    if emulated:
        em = emulated["threads"]["1"]
        ax.bar(
            [x + width / 2 for x in xs],
            [em.get(m, {}).get("p50_ms", 0) for m in models],
            width,
            color=SAND,
            label="arm64 émulé QEMU, 1 thread (p50, indicatif)",
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [m.partition(".")[2] or "fp32" for m in models], rotation=40, ha="right", fontsize=7
    )
    ax.set_ylabel("Latence par image (ms)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
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
                label=tag.replace("_", " "),
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
    ]
    (args.out / "results_tables.md").write_text("\n\n".join(tables) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
