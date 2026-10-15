"""Tests for Week 4: VFI and Howard on the growth model. Run pytest from lecture04/code."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfi import (params, grid, vfi, howard, closed_form_value, closed_form_policy,  # noqa: E402
                 iterations_needed, utility_matrix, bellman, vfi_naive)


@pytest.fixture(scope="module")
def solved():
    p = params()
    return p, vfi(p)


def test_vfi_value_matches_closed_form_on_a_fine_grid():
    p = params(n_k=800)
    s = vfi(p)
    err = np.max(np.abs(s["V"] - closed_form_value(s["k"], p["alpha"], p["beta"])))
    assert err < 0.05


def test_vfi_error_shrinks_with_grid_size():
    errs = []
    for n in (50, 200, 800):
        p = params(n_k=n)
        s = vfi(p)
        errs.append(np.max(np.abs(s["V"] - closed_form_value(s["k"], p["alpha"], p["beta"]))))
    assert errs[0] > errs[1] > errs[2]


def test_policy_within_one_grid_step_of_closed_form(solved):
    p, s = solved
    k = s["k"]
    step = k[1] - k[0]
    interior = (k > 0.05) & (k < 0.45)
    assert np.max(np.abs(k[s["policy"]][interior] - closed_form_policy(k[interior], p["alpha"], p["beta"]))) <= step + 1e-12


def test_distances_shrink_at_rate_beta(solved):
    p, s = solved
    d = s["dists"]
    ratios = d[-20:] / d[-21:-1]
    assert np.allclose(ratios, p["beta"], atol=0.02)


def test_iterations_scale_with_one_over_one_minus_beta():
    n90 = vfi(params(beta=0.90))["n_iter"]
    n99 = vfi(params(beta=0.99))["n_iter"]
    assert n99 > 5 * n90
    assert abs(iterations_needed(0.9, 1e-6, 1.0) - 131) <= 1


def test_howard_gives_the_same_policy_in_far_fewer_steps(solved):
    p, s = solved
    h = howard(p)
    assert np.array_equal(h["policy"], s["policy"])
    assert h["n_iter"] < 15
    assert np.max(np.abs(h["V"] - s["V"])) < 1e-4


def test_aliasing_bug_would_be_caught(solved):
    """A correct VFI changes V on the first iteration; the V_new = V bug reports a distance of exactly zero."""
    p, s = solved
    assert s["dists"][0] > 0 and s["n_iter"] > 1
    # the bug, reproduced on purpose
    k = grid(p)
    U = utility_matrix(k, p["alpha"])
    V = np.zeros(len(k))
    V_new = V                                  # alias, not a copy
    TV, _ = bellman(V, U, p["beta"])
    V_new[:] = TV                              # writes through to V as well
    assert np.max(np.abs(V_new - V)) == 0.0    # so the stopping rule fires immediately


def test_naive_and_vectorised_agree():
    p = params(n_k=40)
    V_naive, n1 = vfi_naive(p)
    s = vfi(p)
    assert np.allclose(V_naive, s["V"], atol=1e-8)
    assert n1 == s["n_iter"]
