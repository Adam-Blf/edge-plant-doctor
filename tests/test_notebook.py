"""The Colab notebook must only call package modules that exist."""

import importlib.util
import json
import re

from edgeplant.config import ROOT

NOTEBOOK = ROOT / "notebooks" / "train_colab.ipynb"


def test_notebook_commands_point_to_real_modules():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    modules = set(re.findall(r"python -m (edgeplant\.\w+)", code))
    assert {
        "edgeplant.data",
        "edgeplant.teacher",
        "edgeplant.train",
        "edgeplant.distill",
    } <= modules
    for module in modules:
        assert importlib.util.find_spec(module) is not None, module


def test_notebook_keeps_outputs_out_of_the_repo():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
