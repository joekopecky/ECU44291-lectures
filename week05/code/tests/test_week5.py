"""Tests for Lecture 5: the theory in numbers, modified policy iteration and policy function iteration. Run pytest from lecture05/code."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfi import (params, grid, vfi, howard, closed_form_value, closed_form_policy,  # noqa: E402
                 iterations_needed, utility_matrix, bellman, modified_policy_iteration)
from cake import value_iteration, policy_iteration  # noqa: E402


@pytest.fixture(scope="module")
def solved():
    p = params()
    return p, vfi(p)


# ---------------------------------------------------------------- the operator
def test_bellman_operator_is_monotone_and_discounts():
    p = params()
    U = utility_matrix(grid(p), p["alpha"])
    rng = np.random.default_rng(0)
    V = rng.normal(size=p["n_k"])
    W = V + np.abs(rng.normal(size=p["n_k"]))        # W >= V pointwise
    TV, _ = bellman(V, U, p["beta"])
    TW, _ = bellman(W, U, p["beta"])
    assert np.all(TW >= TV - 1e-12)                   # Blackwell: monotone
    TVc, _ = bellman(V + 3.0, U, p["beta"])
    assert np.max(np.abs(TVc - (TV + p["beta"] * 3.0))) < 1e-10   # Blackwell: discounts
    assert np.max(np.abs(TV - TW)) <= p["beta"] * np.max(np.abs(V - W)) + 1e-12   # a contraction


def test_first_distance_is_positive_and_ratio_settles_at_beta(solved):
    p, s = solved
    assert s["dists"][0] > 1.0
    late = s["dists"][-5:] / s["dists"][-6:-1]
    assert np.allclose(late, p["beta"], atol=1e-3)


def test_iteration_count_bound_holds(solved):
    p, s = solved
    bound = iterations_needed(p["beta"], 1e-6, s["dists"][0])
    assert s["n_iter"] <= bound <= s["n_iter"] + 5
    assert iterations_needed(0.95, 1e-6, 10) == 315
    assert iterations_needed(0.99, 1e-6, 10) == 1604


def test_vfi_matches_closed_form(solved):
    p, s = solved
    assert np.max(np.abs(s["V"] - closed_form_value(s["k"], p["alpha"], p["beta"]))) < 1e-3
    assert np.max(np.abs(s["k"][s["policy"]] - closed_form_policy(s["k"], p["alpha"], p["beta"]))) < 2 * (s["k"][1] - s["k"][0])


# ---------------------------------------------------------------- the cake by hand
def test_cake_value_iteration_matches_the_board():
    (V1, p1), (V2, p2), (V3, p3) = value_iteration(beta=0.9, n_iter=3)
    assert np.allclose(V1, [0, 1, np.sqrt(2)], atol=1e-3)
    assert np.allclose(V2, [0, 1, 1.9], atol=1e-3)
    assert list(p2) == [0, 0, 1]
    assert np.allclose(V3, V2)


def test_cake_policy_iteration_stops_after_two_steps_at_v2():
    steps = policy_iteration(beta=0.9)
    assert len(steps) == 2
    policy0, r0, v0, improved0 = steps[0]
    assert list(policy0) == [0, 0, 0]
    assert np.allclose(r0, [0, 1, np.sqrt(2)], atol=1e-3)
    assert np.allclose(v0, [0, 1, np.sqrt(2)], atol=1e-3)
    assert list(improved0) == [0, 0, 1]
    policy1, _, v1, improved1 = steps[1]
    assert list(policy1) == [0, 0, 1]
    assert np.allclose(v1, [0, 1, 1.9], atol=1e-3)
    assert list(improved1) == [0, 0, 1]


# ---------------------------------------------------------------- modified policy iteration
def test_modified_policy_iteration_with_m_zero_is_vfi(solved):
    p, s = solved
    r = modified_policy_iteration(p, 0)
    assert r["n_max"] == s["n_iter"] == 132
    assert np.array_equal(r["policy"], s["policy"])
    assert np.array_equal(r["V"], s["V"])


def test_modified_policy_iteration_needs_fewer_maximisations_as_m_grows(solved):
    p, s = solved
    counts = {m: modified_policy_iteration(p, m)["n_max"] for m in (0, 1, 5, 20)}
    assert counts == {0: 132, 1: 66, 5: 23, 20: 8}
    for m in (1, 5, 20):
        assert np.array_equal(modified_policy_iteration(p, m)["policy"], s["policy"])


def test_modified_policy_iteration_with_large_m_reaches_howard(solved):
    p, _ = solved
    h = howard(p)
    r = modified_policy_iteration(p, 500)
    assert r["n_max"] == h["n_iter"] + 1          # one more max to confirm, because the stopping rule is on V
    assert np.array_equal(r["policy"], h["policy"])
    assert np.max(np.abs(r["V"] - h["V"])) < 1e-10



# ---------------------------------------------------------------- Howard on the growth model
def test_howard_same_policy_as_vfi_and_six_steps(solved):
    p, s = solved
    h = howard(p)
    assert h["n_iter"] == 6
    assert np.array_equal(h["policy"], s["policy"])
    gap = round(float(np.max(np.abs(h["V"] - s["V"]))), 4)       # the printed gap: against VFI's own V
    assert gap == 0.0
    gap_cf = round(float(np.max(np.abs(h["V"] - closed_form_value(h["k"], p["alpha"], p["beta"])))), 4)
    assert gap_cf == 0.0001                                        # against the closed form: the grid, not the method


def test_howard_count_does_not_grow_with_beta():
    counts = {b: howard(params(beta=b))["n_iter"] for b in (0.9, 0.99)}
    iters = {b: vfi(params(beta=b))["n_iter"] for b in (0.9, 0.99)}
    assert counts[0.9] == 6 and counts[0.99] == 8
    assert iters[0.9] == 132 and iters[0.99] == 1363


def test_policy_value_satisfies_the_evaluation_identity(solved):
    p, s = solved
    h = howard(p)
    U = utility_matrix(h["k"], p["alpha"])
    n = len(h["policy"])
    r = U[np.arange(n), h["policy"]]
    P = np.zeros((n, n)); P[np.arange(n), h["policy"]] = 1.0
    assert np.max(np.abs(h["V"] - r - p["beta"] * P @ h["V"])) < 1e-10
