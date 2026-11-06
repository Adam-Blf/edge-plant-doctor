import onnx
import torch

from edgeplant.ablation import families
from edgeplant.export import export_onnx
from edgeplant.models import student


def test_families_partition_every_linear_node(tmp_path):
    torch.manual_seed(0)
    checkpoint = tmp_path / "net.pt"
    torch.save(student(pretrained=False).state_dict(), checkpoint)
    path = export_onnx("student", checkpoint, tmp_path / "net.onnx")

    fams = families(path)
    linear = [
        n.name for n in onnx.load(str(path)).graph.node if n.op_type in {"Conv", "Gemm", "MatMul"}
    ]
    flat = [name for names in fams.values() for name in names]

    assert sorted(flat) == sorted(linear)  # every linear node, each exactly once
    assert len(fams["stem"]) == 1
    assert len(fams["depthwise"]) == 11
    assert len(fams["pointwise"]) == 40
    assert fams["classifier"]
