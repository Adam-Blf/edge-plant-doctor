import json

from edgeplant.config import NUM_CLASSES, ROOT
from edgeplant.labels import CLASSES, label_entries


def test_classes_are_the_sorted_dataset_folders():
    assert len(CLASSES) == NUM_CLASSES
    assert list(CLASSES) == sorted(CLASSES)


def test_every_class_has_a_french_name():
    entries = label_entries()
    assert len(entries) == NUM_CLASSES
    assert all(e["crop"] and e["condition"] for e in entries)
    assert sum(e["healthy"] for e in entries) == 12


def test_shipped_labels_match_the_source():
    shipped = json.loads((ROOT / "models" / "labels.json").read_text(encoding="utf-8"))
    assert shipped == label_entries()
