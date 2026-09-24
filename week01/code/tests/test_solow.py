"""Tests for the Lecture 1 Solow module. Run with:  pytest  (from lecture01/code)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solow import fixed_point, k_next, simulate, steady_state  # noqa: E402


def test_steady_state_is_a_fixed_point():
    kstar = steady_state()
    assert abs(k_next(kstar) - kstar) < 1e-12


def test_path_converges_to_steady_state():
    path = simulate(k0=0.25, T=200)
    assert abs(path[-1] - steady_state()) < 1e-6


def test_convergence_from_above_and_below():
    kstar = steady_state()
    assert simulate(0.25, 200)[-1] == pytest.approx(kstar, abs=1e-6)
    assert simulate(3.50, 200)[-1] == pytest.approx(kstar, abs=1e-6)


def test_fixed_point_matches_closed_form():
    x, n_it = fixed_point(k_next, 0.25)
    assert x == pytest.approx(steady_state(), abs=1e-7)
    assert 50 < n_it < 200


def test_fixed_point_fails_loudly():
    with pytest.raises(RuntimeError):
        fixed_point(lambda x: 2 * x + 1, 1.0, max_iter=20)   # slope 2: diverges


def test_convergence_rate_matches_slope():
    alpha, s, n, g, delta = 0.33, 0.30, 0.03, 0.04, 0.15
    slope = 1 - (1 - alpha) * (delta + n + g)
    gap = np.abs(simulate(1.25, 30) - steady_state())
    ratio = gap[21:30] / gap[20:29]          # near k*, gap shrinks by g'(k*) each period
    assert np.allclose(ratio, slope, atol=0.01)
