"""A two-period overlapping-generations economy in general equilibrium.

Week 2 of ECU44291. Households live two periods (young, old), work when young,
save at the after-tax gross return R, and may receive transfers. A government
taxes labour and capital income and can run a pay-as-you-go pension financed by
a payroll contribution tau_p. Firms are Cobb-Douglas. Log utility, so the
household's saving rule is closed-form and the whole equilibrium reduces to
one equation in K.

Solved by root finding on the residual.
"""
import numpy as np
from scipy.optimize import brentq

DEFAULT = dict(alpha=0.3, A=1.0, beta=0.9, delta=0.0, tau_L=0.2, tau_K=0.15,
               t_y=0.0, t_o=0.0, tau_p=0.0, N_y=1.0, N_o=1.0)


def params(**overrides):
    """A parameter dictionary with the module defaults, some overridden."""
    p = dict(DEFAULT)
    p.update(overrides)
    return p


def prices(K, p):
    """Rental rate q, wage w, and the household's after-tax gross return R."""
    L = p["N_y"]
    q = p["alpha"] * p["A"] * K ** (p["alpha"] - 1) * L ** (1 - p["alpha"])
    w = (1 - p["alpha"]) * p["A"] * K ** p["alpha"] * L ** (-p["alpha"])
    R = 1 + (1 - p["tau_K"]) * (q - p["delta"])
    return q, w, R


def pension_benefit(w, p):
    """Pay-as-you-go benefit per old household: contributions of the young, shared out."""
    return p["tau_p"] * w * p["N_y"] / p["N_o"]


def savings(w, R, p):
    """Optimal saving of a young household with log utility.

    max log(c_y) + beta log(c_o)
    c_y + s = (1 - tau_L - tau_p) w + t_y
    c_o     = R s + t_o + pension
    """
    income_y = (1 - p["tau_L"] - p["tau_p"]) * w + p["t_y"]
    income_o = p["t_o"] + pension_benefit(w, p)
    return (p["beta"] * R * income_y - income_o) / ((1 + p["beta"]) * R)


def capital_supply(K, p):
    """Aggregate saving of the young when the capital stock is K."""
    if K <= 0:
        return 0.0                      # no capital, no wage, nothing to save: the trivial equilibrium
    _, w, R = prices(K, p)
    return p["N_y"] * savings(w, R, p)


def residual(K, p):
    """Capital supplied by households minus the capital stock they were given. Zero in equilibrium."""
    return capital_supply(K, p) - K


def solve_by_root(p, bracket=(1e-3, 2.0)):
    """Equilibrium K by Brent's method on the residual. The bracket must exclude K = 0, which is also a root."""
    return brentq(residual, bracket[0], bracket[1], args=(p,), xtol=1e-12)


def steady_state(K, p):
    """Everything else in the economy at capital stock K, plus the aggregate resource check."""
    q, w, R = prices(K, p)
    s = savings(w, R, p)
    Y = p["A"] * K ** p["alpha"] * p["N_y"] ** (1 - p["alpha"])
    r = q - p["delta"]
    c_y = (1 - p["tau_L"] - p["tau_p"]) * w + p["t_y"] - s
    c_o = R * s + p["t_o"] + pension_benefit(w, p)
    C = p["N_y"] * c_y + p["N_o"] * c_o
    G = p["tau_L"] * w * p["N_y"] + p["tau_K"] * r * s * p["N_o"] - p["t_y"] * p["N_y"] - p["t_o"] * p["N_o"]
    arc = Y - p["delta"] * K - C - G          # should be zero in a steady state
    return dict(K=K, Y=Y, q=q, r=r, R=R, w=w, s=s, c_y=c_y, c_o=c_o, C=C, G=G, arc=arc)
