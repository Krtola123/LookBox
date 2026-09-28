"""M6 AI layer: registry, verified download, runtime fallback, BiRefNet processing.

The real BiRefNet can't be downloaded in CI, so:
- downloads are tested for real through file:// URLs;
- the runtime is tested against real onnxruntime with a tiny hand-encoded model,
  and against a fake onnxruntime for the GPU-fallback paths;
- BiRefNet pre/post-processing is tested with a stand-in model function.
"""

import contextlib
import hashlib
import os
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lookbox.ai import birefnet as B
from lookbox.ai import registry as R
from lookbox.ai import runtime as RT
from tests.onnx_tiny import sigmoid_model

# ------------------------------------------------------------------ registry


def test_registry_has_verified_models():
    reg = R.load_registry()
    assert set(reg) >= {"birefnet", "birefnet-lite"}
    full, lite = reg["birefnet"], reg["birefnet-lite"]
    for spec in (full, lite):
        assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0
        assert spec.url.startswith("https://huggingface.co/") and spec.input_size == 1024
        assert spec.mean == (0.485, 0.456, 0.406) and spec.std == (0.229, 0.224, 0.225)
    assert full.size == 972666916 and lite.size == 224005088  # from the LFS pointers
    assert full.size_mb == 973 and lite.size_mb == 224


def _spec_for(tmp_path, payload: bytes, **over):
    src = tmp_path / "remote.onnx"
    src.write_bytes(payload)
    spec = R.ModelSpec(id="t", name="Test", task="background", url=src.as_uri(), source="", filename="t.onnx",
                       sha256=hashlib.sha256(payload).hexdigest(), size=len(payload), input_size=1024,
                       mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return replace(spec, **over)


@contextlib.contextmanager
def _models_dir(tmp_path):
    """Point downloads at a temp folder; always restore the environment."""
    old = os.environ.get("LOOKBOX_MODELS_DIR")
    os.environ["LOOKBOX_MODELS_DIR"] = str(tmp_path / "models")
    try:
        yield
    finally:
        if old is None:
            del os.environ["LOOKBOX_MODELS_DIR"]
        else:
            os.environ["LOOKBOX_MODELS_DIR"] = old


def test_download_verifies_and_marks_ready(tmp_path):
    with _models_dir(tmp_path):
        payload = os.urandom(3 * R.CHUNK + 123)
        spec = _spec_for(tmp_path, payload)
        seen = []
        assert not R.is_ready(spec)
        path = R.download(spec, progress=lambda d, t: seen.append((d, t)))
        assert open(path, "rb").read() == payload and R.is_ready(spec)
        assert seen[-1] == (len(payload), len(payload)) and len(seen) == 4
        assert not os.path.exists(path + ".part")


def test_corrupted_download_is_rejected_and_leaves_nothing(tmp_path):
    with _models_dir(tmp_path):
        spec = _spec_for(tmp_path, b"x" * 1000, sha256="0" * 64)
        with pytest.raises(R.DownloadError, match="integrity"):
            R.download(spec)
        assert not os.path.exists(R.model_path(spec)) and not os.path.exists(R.model_path(spec) + ".part")
        assert not R.is_ready(spec)


def test_short_and_oversized_downloads_are_rejected(tmp_path):
    with _models_dir(tmp_path):
        spec = _spec_for(tmp_path, b"y" * 1000)
        with pytest.raises(R.DownloadError, match="incomplete"):
            R.download(replace(spec, size=2000))
        with pytest.raises(R.DownloadError, match="more data"):
            R.download(replace(spec, size=500))
        assert not os.path.exists(R.model_path(spec))


def test_download_cancel_and_unreachable(tmp_path):
    with _models_dir(tmp_path):
        spec = _spec_for(tmp_path, os.urandom(4 * R.CHUNK))
        ev = threading.Event()
        with pytest.raises(R.DownloadCancelled):
            R.download(spec, progress=lambda d, t: ev.set(), cancel=ev)
        assert not os.path.exists(R.model_path(spec) + ".part")
        with pytest.raises(R.DownloadError, match="reach"):
            R.download(replace(spec, url=(tmp_path / "nope.onnx").as_uri()))


def test_is_ready_needs_the_marker_and_the_right_size(tmp_path):
    with _models_dir(tmp_path):
        payload = b"z" * 2048
        spec = _spec_for(tmp_path, payload)
        R.download(spec)
        assert R.is_ready(spec)
        with open(R.model_path(spec), "ab") as fh:
            fh.write(b"!")  # file tampered with / truncated later
        assert not R.is_ready(spec)


# ------------------------------------------------------------------ runtime (real onnxruntime)


def test_real_onnxruntime_session_and_run(tmp_path):
    path = tmp_path / "tiny.onnx"
    path.write_bytes(sigmoid_model(16))
    mgr = RT.ModelManager()
    assert mgr.input_size(str(path), default=1024) == 16
    out = mgr.run(str(path), np.zeros((1, 3, 16, 16), np.float32))
    assert out.shape == (1, 3, 16, 16) and np.allclose(out, 0.5)
    assert mgr.provider in (RT.GPU, RT.CPU)


class _FakeSession:
    def __init__(self, provider, fail_run=False):
        self.provider, self.fail_run = provider, fail_run

    def get_providers(self):
        return [self.provider]

    def get_inputs(self):
        return [SimpleNamespace(name="input_image", shape=[1, 3, 1024, 1024])]

    def run(self, _outs, feeds):
        if self.fail_run:
            raise RuntimeError("out of GPU memory")
        return [feeds["input_image"] * 2]


def _fake_ort(available, dml_create_fails=False, dml_run_fails=False):
    created = []

    def session(path, sess_options=None, providers=None):
        created.append(providers)
        if providers[0] == RT.GPU:
            if dml_create_fails:
                raise RuntimeError("DirectML device not found")
            return _FakeSession(RT.GPU, fail_run=dml_run_fails)
        return _FakeSession(RT.CPU)

    ort = SimpleNamespace(SessionOptions=lambda: SimpleNamespace(), ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
                          get_available_providers=lambda: available, InferenceSession=session)
    return ort, created


def test_gpu_used_when_available():
    ort, created = _fake_ort([RT.GPU, RT.CPU])
    s, provider, notice = RT.create_session("m.onnx", ort=ort)
    assert provider == RT.GPU and notice is None and created == [[RT.GPU, RT.CPU]]


def test_gpu_session_failure_falls_back_to_cpu_with_notice():
    ort, created = _fake_ort([RT.GPU, RT.CPU], dml_create_fails=True)
    s, provider, notice = RT.create_session("m.onnx", ort=ort)
    assert provider == RT.CPU and "GPU unavailable" in notice
    ort, _ = _fake_ort([RT.CPU])
    assert "No GPU runtime" in RT.create_session("m.onnx", ort=ort)[2]


def test_gpu_run_failure_retries_on_cpu_once_and_stays_there():
    ort, created = _fake_ort([RT.GPU, RT.CPU], dml_run_fails=True)
    mgr = RT.ModelManager(ort=ort)
    out = mgr.run("m.onnx", np.ones((1, 3, 2, 2), np.float32))
    assert np.allclose(out, 2) and mgr.provider == RT.CPU and "retried on the CPU" in mgr.notice
    mgr.run("m.onnx", np.ones((1, 3, 2, 2), np.float32))
    assert len(created) == 2  # no GPU retry loop on every run


# ------------------------------------------------------------------ BiRefNet processing


SPEC = R.load_registry()["birefnet"]


def test_preprocess_matches_the_published_config():
    rgba = np.zeros((10, 20, 4), np.float32)
    rgba[:, :, :3] = (0.485, 0.456, 0.406)  # exactly the mean → normalises to 0
    rgba[:, :, 3] = 1
    x = B.preprocess(rgba, 32, SPEC.mean, SPEC.std)
    assert x.shape == (1, 3, 32, 32) and x.dtype == np.float32 and np.abs(x).max() < 1e-5
    white = np.ones((8, 8, 4), np.float32)
    x = B.preprocess(white, 8, SPEC.mean, SPEC.std)
    assert np.allclose(x[0, 0], (1 - 0.485) / 0.229, atol=1e-5)  # channel order R, G, B
    assert np.allclose(x[0, 2], (1 - 0.406) / 0.225, atol=1e-5)


def test_postprocess_handles_logits_and_probabilities():
    probs = np.full((1, 1, 4, 4), 0.25, np.float32)
    assert np.allclose(B.postprocess(probs, 8, 8), 0.25)
    logits = np.full((1, 1, 4, 4), 3.0, np.float32)
    assert np.allclose(B.postprocess(logits, 8, 8), 1 / (1 + np.exp(-3.0)), atol=1e-6)
    assert B.postprocess(np.zeros((4, 4), np.float32), 3, 5).shape == (3, 5)


def test_remove_background_end_to_end_with_a_stand_in_model():
    h, w = 90, 160
    rgba = np.zeros((h, w, 4), np.float32)
    rgba[:, :, 3] = 1.0
    rgba[20:70, 50:110, :3] = 0.9  # the "subject"
    rgba[:, :5, 3] = 0.0  # a strip that was already transparent

    def fake_model(x):  # logits: +8 where the (normalised) input is bright, −8 elsewhere
        assert x.shape == (1, 3, 64, 64)
        return np.where(x[:, :1] > 1.0, 8.0, -8.0).astype(np.float32)

    mask = B.remove_background(fake_model, rgba, SPEC, size=64)
    assert mask.shape == (h, w) and mask.dtype == np.float32
    assert mask[45, 80] > 0.99 and mask[5, 20] < 0.01  # subject kept, background gone
    assert mask[:, :5].max() == 0.0  # already-transparent stays hidden
    refined = B.remove_background(fake_model, rgba, SPEC, size=64, refine=True)
    assert refined.shape == (h, w) and refined[45, 80] > 0.9


def test_real_runtime_drives_remove_background(tmp_path):
    path = tmp_path / "tiny.onnx"
    path.write_bytes(sigmoid_model(32))
    mgr = RT.ModelManager()
    rgba = np.random.default_rng(0).random((50, 70, 4), dtype=np.float32)
    rgba[:, :, 3] = 1
    mask = B.remove_background(lambda x: mgr.run(str(path), x), rgba, SPEC,
                               size=mgr.input_size(str(path), 1024))
    assert mask.shape == (50, 70) and 0 <= mask.min() and mask.max() <= 1
