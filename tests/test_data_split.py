import numpy as np
import pytest

from edgeplant.config import NUM_CLASSES
from edgeplant.data import Item, leaf_group, split_manifest

CAPS = {"train": 6, "val": 2, "test": 3}


def synthetic_items(per_class: int = 40, leaf_size: int = 4) -> list[Item]:
    items = []
    for c in range(NUM_CLASSES):
        for k in range(per_class):
            leaf = f"leaf_{c}_{k // leaf_size}"
            items.append(Item(f"raw/color/class{c:02d}/img{k}.jpg", f"class{c:02d}", leaf))
    return items


def test_leaf_group_uses_map_and_falls_back():
    leaf_map = {"mary_hl 9155": ["Raspberry___healthy:::12.0"]}
    name = "a37f34f7-022d-461a-8a3d-95f5cd774e35___Mary_HL 9155.JPG"
    assert leaf_group(name, "Raspberry___healthy", leaf_map) == "Raspberry___healthy:::12.0"
    assert leaf_group("x___Unknown 1.JPG", "A", leaf_map) == "single_Unknown 1"


def test_split_never_shares_a_leaf_between_splits():
    splits = split_manifest(synthetic_items(), CAPS)
    leaves = {name: {i.group for i in items} for name, items in splits.items()}
    assert not leaves["train"] & leaves["test"]
    assert not leaves["train"] & leaves["val"]
    assert not leaves["val"] & leaves["test"]


def test_split_is_capped_and_covers_all_classes():
    splits = split_manifest(synthetic_items(), CAPS)
    for name, cap in CAPS.items():
        counts = np.bincount([int(i.label[-2:]) for i in splits[name]], minlength=NUM_CLASSES)
        assert counts.max() <= cap and counts.min() >= 1


def test_split_is_deterministic():
    items = synthetic_items()
    assert split_manifest(items, CAPS) == split_manifest(items, CAPS)


def test_wrong_class_count_is_rejected():
    with pytest.raises(ValueError):
        split_manifest(synthetic_items()[:100], CAPS)
