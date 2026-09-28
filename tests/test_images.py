import os

import lookbox  # noqa: F401
import cv2
import numpy as np
import pytest

from lookbox.core.io import images
from tests.helpers import make_rgba, write_png


def _roundtrip(path):
    with open(path, "rb") as fh:
        return images.decode(fh.read(), os.path.splitext(path)[1])


def test_decode_rgba8_exact(tmp_path):
    src = make_rgba(7, 5, seed=1)
    out = _roundtrip(write_png(str(tmp_path / "a.png"), src))
    assert out.dtype == np.float32 and out.shape == (5, 7, 4)
    assert np.array_equal(out, src)


def test_decode_rgba16_keeps_precision(tmp_path):
    src = np.linspace(0, 1, 4 * 3 * 4, dtype=np.float32).reshape(3, 4, 4)
    out = _roundtrip(write_png(str(tmp_path / "a.png"), src, bits=16))
    assert np.abs(out - src).max() <= 0.5 / 65535 + 1e-7


def test_decode_gray_and_rgb_get_opaque_alpha(tmp_path):
    gray = (np.arange(12, dtype=np.uint8) * 20).reshape(3, 4)
    p = str(tmp_path / "g.png")
    cv2.imwrite(p, gray)
    out = _roundtrip(p)
    assert np.allclose(out[:, :, 0], gray / 255.0) and np.all(out[:, :, 3] == 1.0)
    assert np.array_equal(out[:, :, 0], out[:, :, 2])

    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[:, :, 2] = 255  # red in BGR
    p = str(tmp_path / "r.jpg")
    cv2.imwrite(p, bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
    out = _roundtrip(p)
    assert out[0, 0, 0] > 0.9 and out[0, 0, 2] < 0.1 and out[0, 0, 3] == 1.0


def test_decode_exr_applies_srgb_curve(tmp_path):
    if not cv2.haveImageWriter(".exr"):
        pytest.skip("OpenCV build has no EXR codec")
    lin = np.full((2, 2, 4), 0.5, np.float32)  # BGRA, linear 0.5
    lin[:, :, 3] = 1.0
    p = str(tmp_path / "a.exr")
    cv2.imwrite(p, lin)
    out = _roundtrip(p)
    assert abs(out[0, 0, 0] - 0.7353569) < 1e-3  # sRGB OETF(0.5)
    assert out[0, 0, 3] == 1.0


def test_bad_bytes_raise_image_error():
    with pytest.raises(images.ImageError):
        images.decode(b"not an image", ".png")


def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.gif"
    p.write_bytes(b"GIF89a")
    with pytest.raises(images.ImageError):
        images.read_file(str(p))


def test_quantize8_exact_on_grid_and_deterministic():
    grid = make_rgba(64, 64, seed=3)
    q = images.quantize8(grid)
    assert np.array_equal(q.astype(np.float32) / 255.0, grid)

    ramp = np.tile(np.linspace(0.2, 0.25, 512, dtype=np.float32)[None, :, None], (64, 1, 4))
    a, b = images.quantize8(ramp), images.quantize8(ramp)
    assert np.array_equal(a, b)
    # Dither preserves the average level (that's what kills banding).
    assert abs(a[:, :, 0].mean() / 255.0 - ramp[:, :, 0].mean()) < 1e-3


def test_encode_png_roundtrip(tmp_path):
    src = make_rgba(9, 6, seed=4)
    p = str(tmp_path / "o.png")
    images.save_png(p, src)
    assert np.array_equal(_roundtrip(p), src)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
