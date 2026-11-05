import numpy as np
from PIL import Image

from edgeplant.preprocess import (
    area_weights,
    center_crop_square,
    normalize,
    preprocess,
    resize_area,
    to_uint8_square,
)


def test_area_weights_rows_sum_to_one():
    for src, dst in [(256, 224), (1080, 224), (224, 224), (100, 224)]:
        w = area_weights(src, dst)
        assert w.shape == (dst, src)
        np.testing.assert_allclose(w.sum(axis=1), 1.0)


def test_constant_image_stays_constant():
    img = np.full((300, 300, 3), 77, dtype=np.uint8)
    np.testing.assert_allclose(resize_area(img), 77.0, atol=1e-4)


def test_center_crop_is_square_and_centered():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[:, 100:200] = 255
    crop = center_crop_square(img)
    assert crop.shape == (100, 100, 3)
    assert crop.min() == 255


def test_integer_downscale_matches_pil_box():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(448, 448, 3), dtype=np.uint8)
    ref = np.asarray(Image.fromarray(img).resize((224, 224), Image.Resampling.BOX))
    ours = to_uint8_square(img)
    assert np.abs(ours.astype(int) - ref.astype(int)).max() <= 1


def test_preprocess_shape_dtype_and_range():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    x = preprocess(img)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert x.min() > -2.2 and x.max() < 2.7


def test_normalize_uses_imagenet_statistics():
    white = np.full((224, 224, 3), 255.0, dtype=np.float32)
    x = normalize(white)
    np.testing.assert_allclose(x[0, :, 0, 0], [2.2489, 2.4286, 2.6400], atol=1e-3)
