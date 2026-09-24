"""The Solow model in efficiency units.

Lecture 1 live-coding target. This is the shape every model in the module
takes: a module with small, documented, testable functions.

    k_{t+1} = s k_t^alpha + (1 - delta - n - g) k_t
"""
import numpy as np


def k_next(k, alpha=0.33, s=0.30, n=0.03, g=0.04, delta=0.15):
    """Next period's capital per effective worker."""
    return s * k**alpha + (1 - delta - n - g) * k


def steady_state(alpha=0.33, s=0.30, n=0.03, g=0.04, delta=0.15):
    """Closed-form steady state k*."""
    return (s / (delta + n + g)) ** (1 / (1 - alpha))


def simulate(k0, T, **params):
    """Path of k from k0 for T periods (an array of length T + 1)."""
    path = [k0]
    for _ in range(T):
        path.append(k_next(path[-1], **params))
    return np.array(path)


def fixed_point(g, x0, tol=1e-8, max_iter=1000):
    """Iterate x <- g(x) from x0 until |g(x) - x| < tol.

    Returns (x, iterations). Raises RuntimeError if max_iter is reached,
    because an algorithm that fails quietly is worse than one that does not run.
    """
    x = x0
    for it in range(max_iter):          # the loop
        x_new = g(x)                    # the update
        if abs(x_new - x) < tol:        # the stopping rule
            return x_new, it + 1
        x = x_new
    raise RuntimeError(f"no convergence after {max_iter} iterations")
