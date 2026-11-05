"""PlantVillage subset: leaf-grouped stratified split, cached as 224x224 uint8 arrays.

The 54k PlantVillage images contain several photos of each physical leaf. A random
split leaks leaves between train and test and inflates accuracy, so images are
grouped by leaf (using the dataset's leaf map) before splitting.
"""

import argparse
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .config import CACHE_DIR, DATA_DIR, NUM_CLASSES, SEED
from .preprocess import to_uint8_square

REPO_ID = "mohanty/PlantVillage"
PER_CLASS = {"train": 150, "val": 40, "test": 100}


@dataclass(frozen=True)
class Item:
    path: str
    label: str
    group: str


def leaf_group(name: str, label: str, leaf_map: dict) -> str:
    """Leaf identifier of an image file, following the dataset's own grouping rules."""
    ident = name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    ident = re.sub(r"\.(jpg|png)$", "", ident, flags=re.IGNORECASE).strip()
    options = leaf_map.get(ident.lower())
    if options is None:
        return f"single_{ident}"
    for option in options:
        if len(options) == 1 or label in option:
            return option
    return f"single_{ident}"


def build_manifest(zip_path: Path, leaf_map_path: Path) -> list[Item]:
    leaf_map = json.loads(leaf_map_path.read_text(encoding="utf-8"))
    items = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) == 4 and parts[1] == "color" and parts[3]:
                items.append(Item(name, parts[2], leaf_group(parts[3], parts[2], leaf_map)))
    return sorted(items, key=lambda i: i.path)


def split_manifest(items: list[Item], caps: dict[str, int | None]) -> dict[str, list[Item]]:
    """Leaf-grouped, class-stratified train/val/test, then capped per class."""
    classes = sorted({i.label for i in items})
    if len(classes) != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} classes, found {len(classes)}")
    y = np.array([classes.index(i.label) for i in items])
    groups = np.array([i.group for i in items])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(y, y, groups))
    picks = {
        "test": folds[0][1],
        "val": folds[1][1],
        "train": np.concatenate([f[1] for f in folds[2:]]),
    }
    rng = np.random.default_rng(SEED)
    out = {}
    for name, idx in picks.items():
        chosen = []
        for c in range(NUM_CLASSES):
            members = idx[y[idx] == c]
            chosen.extend(rng.permutation(members)[: caps[name]])
        out[name] = [items[i] for i in sorted(chosen)]
    return out


def build_cache(
    zip_path: Path, leaf_map_path: Path, out_dir: Path = CACHE_DIR, full: bool = False
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = build_manifest(zip_path, leaf_map_path)
    splits = split_manifest(items, {k: None for k in PER_CLASS} if full else PER_CLASS)
    classes = sorted({i.label for i in items})
    summary = {"classes": classes, "counts": {}, "leaf_groups_shared": 0}
    leaves = {k: {i.group for i in v} for k, v in splits.items()}
    summary["leaf_groups_shared"] = sum(
        len(leaves[a] & leaves[b])
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    )
    with zipfile.ZipFile(zip_path) as z:
        for name, subset in splits.items():
            x = np.zeros((len(subset), 224, 224, 3), dtype=np.uint8)
            for k, item in enumerate(subset):
                rgb = np.asarray(Image.open(io.BytesIO(z.read(item.path))).convert("RGB"))
                x[k] = to_uint8_square(rgb)
            np.save(out_dir / f"{name}_x.npy", x)
            np.save(out_dir / f"{name}_y.npy", np.array([classes.index(i.label) for i in subset]))
            (out_dir / f"{name}_files.txt").write_text(
                "\n".join(i.path for i in subset), encoding="utf-8"
            )
            summary["counts"][name] = len(subset)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def load_split(name: str, cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(cache_dir / f"{name}_x.npy", mmap_mode="r")
    return x, np.load(cache_dir / f"{name}_y.npy")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the cached PlantVillage subset")
    ap.add_argument(
        "--zip", type=Path, help="data.zip of mohanty/PlantVillage (downloaded if absent)"
    )
    ap.add_argument("--leaf-map", type=Path)
    ap.add_argument("--full", action="store_true", help="no per-class cap (Colab GPU run)")
    args = ap.parse_args()
    if args.zip is None:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(REPO_ID, repo_type="dataset", local_dir=DATA_DIR / "raw"))
        args.zip, args.leaf_map = root / "data.zip", root / "leaf_grouping" / "leaf-map.json"
    print(json.dumps(build_cache(args.zip, args.leaf_map, full=args.full), indent=1)[-600:])


if __name__ == "__main__":
    main()
