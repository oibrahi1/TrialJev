"""The common development-fitted correction: the test decision must resolve exact ties exactly as the development fit does."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import calibrate_all as ca  # noqa: E402


def test_exact_ties_count_as_excluded_in_fit_and_in_test_decision():
    # perfectly separable development scores: the first grid point already reaches 100%, so the fit keeps weights (1, 1) and an exact
    # tie at p = 0.5 survives the correction; it must be called excluded, as it is inside the fit (argmax picks the first class)
    p_dev = np.array([0.9, 0.1, 0.9, 0.1])
    y_dev = np.array([1, 0, 1, 0])
    T, w, _ = ca.fit(np.stack([1 - p_dev, p_dev], 1), y_dev)
    assert np.allclose(w, 1.0)
    p_test = np.array([0.5, 0.5, 0.9, 0.1])
    q, decision, info = ca.correct(p_dev, y_dev, p_test)
    assert q[0] == 0.5 and q[1] == 0.5
    assert not decision[0] and not decision[1]
    assert decision[2] and not decision[3]


def test_correction_changes_only_the_decision_threshold_not_the_order():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 200)
    y = (p + rng.normal(0, 0.2, 200)) > 0.4
    q, decision, info = ca.correct(p, y.astype(int), p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)          # monotonic
    assert info["T"] > 0 and info["w_true"] > 0 and info["w_false"] > 0
