"""A life-cycle household solved by backward induction on a grid.

Week 3 of ECU44291. A household lives J periods, works for the first J_R,
earns w times an age-efficiency profile, pays a contribution rate tau on
earnings, receives a pension when retired, saves at gross return R, cannot
borrow, and starts and ends life with no assets. Choices of next-period
assets are restricted to the grid (continuous choice comes in Week 5).

This is the shape of M1 Part A.
"""
import numpy as np

DEFAULT = dict(J=60, J_R=45, beta=0.96, R=1.04, sigma=2.0, w=1.0, tau=0.0,
               repl=0.4, a_max=25.0, n_a=300)


def params(**overrides):
    p = dict(DEFAULT)
    p.update(overrides)
    return p


def efficiency_profile(J, J_R):
    """Hump-shaped age-efficiency units, normalised to mean one over working life."""
    j = np.arange(1, J + 1)
    e = np.exp(0.06 * j - 0.0012 * j**2)
    e = e / e[:J_R].mean()
    e[J_R:] = 0.0
    return e


def income_profile(p):
    """After-contribution labour income while working, pension when retired. Length J."""
    e = efficiency_profile(p["J"], p["J_R"])
    y = (1 - p["tau"]) * p["w"] * e
    pension = p["repl"] * p["w"] * e[:p["J_R"]].mean()
    y[p["J_R"]:] = pension
    return y


def utility(c, sigma):
    c = np.asarray(c, dtype=float)
    if sigma == 1.0:
        return np.log(c)
    return c ** (1 - sigma) / (1 - sigma)


def solve_household_grid(p):
    """Backward induction with choices on the grid.

    Returns dict with a_grid, V (J+1 by n_a; row J is the terminal zero), policy
    (J by n_a, index of next-period assets), and the income profile y.
    """
    J, R, beta, sigma = p["J"], p["R"], p["beta"], p["sigma"]
    a = np.linspace(0.0, p["a_max"], p["n_a"])
    y = income_profile(p)
    V = np.zeros((J + 1, len(a)))            # V[J] = 0: nothing after death
    policy = np.zeros((J, len(a)), dtype=int)
    for j in range(J - 1, -1, -1):           # backward: last age first
        c = R * a[:, None] + y[j] - a[None, :]          # rows: today's assets, columns: choices
        feasible = c > 0
        u = np.full(c.shape, -np.inf)
        u[feasible] = utility(c[feasible], sigma)
        value = u + beta * V[j + 1][None, :]
        policy[j] = value.argmax(axis=1)
        V[j] = value.max(axis=1)
    return dict(a_grid=a, V=V, policy=policy, y=y)


def simulate_profile(sol, p, a0_index=0):
    """Follow the policy from an initial asset index. Returns (assets of length J+1, consumption of length J)."""
    a, policy, y, R = sol["a_grid"], sol["policy"], sol["y"], p["R"]
    idx = a0_index
    assets = [a[idx]]
    cons = []
    for j in range(p["J"]):
        nxt = policy[j, idx]
        cons.append(R * a[idx] + y[j] - a[nxt])
        assets.append(a[nxt])
        idx = nxt
    return np.array(assets), np.array(cons)


def two_period_closed_form(y1, y2, R, beta):
    """Log utility, two periods, no borrowing: c1 = (y1 + y2/R)/(1+beta), s = y1 - c1 (or 0 if that is negative)."""
    c1 = (y1 + y2 / R) / (1 + beta)
    s = max(y1 - c1, 0.0)
    c1 = y1 - s
    c2 = R * s + y2
    return dict(c1=c1, c2=c2, s=s)


