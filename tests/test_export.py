import numpy as np
import onnxruntime as ort
import torch

from edgeplant.export import export_onnx
from edgeplant.models import student


def test_onnx_export_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    net = student(pretrained=False).eval()
    checkpoint = tmp_path / "net.pt"
    torch.save(net.state_dict(), checkpoint)
    path = export_onnx("student", checkpoint, tmp_path / "net.onnx")
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        expected = net(x).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": x.numpy()})[0]
    assert actual.shape == (3, 38)
    np.testing.assert_allclose(actual, expected, atol=1e-4)
    assert (actual.argmax(1) == expected.argmax(1)).all()


def test_every_quantized_variant_loads_and_runs(tmp_path):
    from onnxruntime.quantization import CalibrationMethod

    from edgeplant.quantize import quantize_all

    torch.manual_seed(0)
    checkpoint = tmp_path / "net.pt"
    torch.save(student(pretrained=False).state_dict(), checkpoint)
    fp32 = export_onnx("student", checkpoint, tmp_path / "net.onnx")
    images = np.random.default_rng(0).integers(0, 256, size=(6, 224, 224, 3), dtype=np.uint8)
    variants = quantize_all(
        fp32, tmp_path, images, {"static_mixed_minmax": CalibrationMethod.MinMax}
    )
    assert set(variants) == {
        "dynamic_gemm",
        "dynamic_all",
        "dynamic_mixed",
        "static_all_minmax",
        "static_mixed_minmax",
    }
    x = np.random.default_rng(1).normal(size=(1, 3, 224, 224)).astype(np.float32)
    for path in variants.values():
        out = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
            None, {"input": x}
        )[0]
        assert out.shape == (1, 38) and np.isfinite(out).all()


def test_float_islands_are_the_stem_and_the_depthwise_convolutions(tmp_path):
    from edgeplant.quantize import float_islands

    torch.manual_seed(0)
    checkpoint = tmp_path / "net.pt"
    torch.save(student(pretrained=False).state_dict(), checkpoint)
    names = float_islands(export_onnx("student", checkpoint, tmp_path / "net.onnx"))
    assert len(names) == 12 and names[0].endswith("features.0/features.0.0/Conv")
