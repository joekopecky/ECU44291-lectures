"""Root finding from first principles: bisection and Newton.

Week 2 of ECU44291. Each solver returns (root, log) where log records the
iterates, so convergence can be plotted and compared. Each fails loudly.
"""
import numpy as np


def bisection(f, a, b, tol=1e-8, max_iter=200):
    """Find a root of f on [a, b] by bisection.

    Requires f(a) and f(b) of opposite sign. Returns (root, brackets) where
    brackets is the list of (a, b) intervals visited. Converges at a linear
    rate: the bracket halves every iteration.
    """
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, [(a, b)]
    if fb == 0:
        return b, [(a, b)]
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs: no bracket")
    brackets = []
    for _ in range(max_iter):                    # the loop
        brackets.append((a, b))
        m = 0.5 * (a + b)                        # the update
        fm = f(m)
        if fm == 0 or 0.5 * (b - a) < tol:       # the stopping rule
            return m, brackets
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    raise RuntimeError(f"bisection: no convergence after {max_iter} iterations")


def newton(f, x0, fprime=None, tol=1e-10, max_iter=50, h=1e-6):
    """Find a root of f by Newton's method from x0.

    fprime is the derivative; if None, a central finite difference with step h
    is used. Returns (root, iterates). Converges quadratically near a simple
    root; may diverge, cycle, or leave the domain from a bad start, in which
    case it raises RuntimeError.
    """
    x = x0
    iterates = [x0]
    for _ in range(max_iter):
        fx = f(x)
        d = fprime(x) if fprime is not None else (f(x + h) - f(x - h)) / (2 * h)
        if not np.isfinite(fx) or not np.isfinite(d):
            raise RuntimeError(f"newton: left the domain at x = {float(x):.4g} (f or f' not finite)")
        if d == 0:
            raise RuntimeError(f"newton: zero derivative at x = {float(x):.4g}")
        x_new = x - fx / d
        iterates.append(x_new)
        if abs(x_new - x) < tol:
            return x_new, iterates
        x = x_new
    raise RuntimeError(f"newton: no convergence after {max_iter} iterations")


# --- the running example: the Solow residual ---------------------------------

def solow_residual(k, alpha=0.33, s=0.30, n=0.03, g=0.04, delta=0.15):
    """g(k) - k for the Solow model of Week 1. Zero at k = 0 and at k*."""
    return s * np.power(k, alpha) - (delta + n + g) * k


def solow_residual_prime(k, alpha=0.33, s=0.30, n=0.03, g=0.04, delta=0.15):
    return alpha * s * np.power(k, alpha - 1) - (delta + n + g)


def solow_kstar(alpha=0.33, s=0.30, n=0.03, g=0.04, delta=0.15):
    return (s / (delta + n + g)) ** (1 / (1 - alpha))
