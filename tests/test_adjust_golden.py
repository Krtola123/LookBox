"""Golden tests: freeze the exact look of every Adjust control (ARCHITECTURE §14).

A failure here means a control now looks different. If that's intended (you
retuned it), regenerate with `pytest --update-golden` and say so in the commit.
"""

import os

import numpy as np
import pytest

from tests import golden_adjust as G


@pytest.mark.parametrize("control", G.CONTROLS)
def test_adjust_golden(control):
    if os.environ.get("LOOKBOX_UPDATE_GOLDEN") == "1":
        os.makedirs(G.GOLDEN_DIR, exist_ok=True)
        np.savez_compressed(G.path(control), **{k: v.astype(np.float16) for k, v in G.compute(control).items()})
        pytest.skip("golden updated")
    if not os.path.exists(G.path(control)):
        pytest.fail(f"Missing {G.path(control)}. Create it with: pytest --update-golden")
    expected = np.load(G.path(control))
    actual = G.compute(control)
    assert sorted(expected.files) == sorted(actual)
    for key, arr in actual.items():
        diff = np.abs(arr - expected[key].astype(np.float32)).max()
        assert diff <= G.TOLERANCE, f"{control} / {key}: max diff {diff:.4f} (look changed)"
