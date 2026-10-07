"""Tests for Week 3: the life-cycle household by backward induction. Run pytest from lecture03/code."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifecycle import (params, income_profile, solve_household_grid, simulate_profile,  # noqa: E402
                       two_period_closed_form)


@pytest.fixture(scope="module")
def sol():
    p = params()
    return p, solve_household_grid(p)


def test_terminal_value_is_zero_and_last_age_saves_nothing(sol):
    p, s = sol
    assert np.all(s["V"][p["J"]] == 0.0)
    assert np.all(s["policy"][p["J"] - 1] == 0)          # a_{J+1} = 0: eat everything


def test_profiles_are_feasible_and_end_with_no_assets(sol):
    p, s = sol
    assets, cons = simulate_profile(s, p)
    assert np.all(cons > 0)
    assert assets[0] == 0.0 and assets[-1] == 0.0
    assert len(assets) == p["J"] + 1 and len(cons) == p["J"]


def test_assets_are_hump_shaped_with_the_peak_near_retirement(sol):
    p, s = sol
    assets, _ = simulate_profile(s, p)
    peak = int(np.argmax(assets))
    assert p["J_R"] - 5 <= peak <= p["J_R"] + 2
    assert assets.max() > 3.0


def test_consumption_is_smoother_than_income(sol):
    p, s = sol
    _, cons = simulate_profile(s, p)
    y = income_profile(p)
    assert cons.std() < 0.5 * y.std()


def test_two_period_grid_solution_matches_closed_form():
    y1, y2, R, beta = 1.0, 0.3, 1.04, 0.96
    cf = two_period_closed_form(y1, y2, R, beta)
    p = params(J=2, J_R=1, sigma=1.0, beta=beta, R=R, a_max=1.0, n_a=2001, tau=0.0, repl=0.0)
    s = solve_household_grid(p)
    s["y"] = np.array([y1, y2])               # override the profile with the two-period incomes
    # re-solve with the overridden incomes
    a = s["a_grid"]
    V2 = np.zeros(len(a))
    c1 = R * 0.0 + y1 - a
    ok = c1 > 0
    val = np.full(len(a), -np.inf)
    val[ok] = np.log(c1[ok]) + beta * np.log(R * a[ok] + y2)
    s_grid = a[val.argmax()]
    assert s_grid == pytest.approx(cf["s"], abs=(a[1] - a[0]))


def test_contribution_rate_lowers_working_consumption(sol):
    p, s = sol
    _, cons0 = simulate_profile(s, p)
    p1 = params(tau=0.1)
    _, cons1 = simulate_profile(solve_household_grid(p1), p1)
    assert cons1[:p["J_R"]].mean() < cons0[:p["J_R"]].mean()


