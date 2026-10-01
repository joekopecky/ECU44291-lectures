"""Tests for Week 2: root finding and the two-period OLG. Run with pytest from lecture02/code."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rootfinding import bisection, newton, solow_residual, solow_residual_prime, solow_kstar  # noqa: E402
from olg2 import params, residual, solve_by_root, steady_state  # noqa: E402

KSTAR = solow_kstar()


def test_bisection_finds_solow_steady_state():
    root, brackets = bisection(solow_residual, 0.5, 4.0)
    assert root == pytest.approx(KSTAR, abs=1e-7)
    assert 20 < len(brackets) < 40          # linear rate: log2(3.5 / 1e-8) = 28.4, so 29 steps


def test_bisection_needs_a_bracket():
    with pytest.raises(ValueError):
        bisection(solow_residual, 2.0, 4.0)  # both negative


def test_newton_converges_fast_from_a_good_start():
    root, its = newton(solow_residual, 3.5, fprime=solow_residual_prime)
    assert root == pytest.approx(KSTAR, abs=1e-9)
    assert len(its) < 10


def test_newton_numerical_derivative_agrees():
    root, _ = newton(solow_residual, 3.5)
    assert root == pytest.approx(KSTAR, abs=1e-7)


def test_newton_leaves_the_domain_from_a_bad_start():
    with pytest.raises(RuntimeError):
        with np.errstate(invalid="ignore"):
            newton(solow_residual, 0.1, fprime=solow_residual_prime)


def test_secant_converges():
    from rootfinding import secant
    root, its = secant(solow_residual, 3.0, 3.5)
    assert root == pytest.approx(KSTAR, abs=1e-8)
    assert len(its) < 12


def test_olg_resource_constraint_holds():
    p = params()
    ss = steady_state(solve_by_root(p), p)
    assert abs(ss["arc"]) < 1e-10
    assert ss["c_y"] > 0 and ss["c_o"] > 0


def test_zero_is_also_a_root():
    p = params()
    assert residual(0.0, p) == 0.0           # w(0) = 0, so nobody saves: the trivial equilibrium


def test_payg_pension_lowers_capital():
    K0 = solve_by_root(params())
    K1 = solve_by_root(params(tau_p=0.2))
    assert K1 < K0
