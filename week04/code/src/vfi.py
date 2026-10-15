"""Value function iteration and Howard's improvement on the neoclassical growth model.

Week 4 of ECU44291. Log utility, Cobb-Douglas production, full depreciation,
choices restricted to the capital grid. This case has a closed form,

    V(k) = A + B ln k,   g(k) = alpha beta k^alpha,

which is the check for everything numerical here.
"""
import time

import numpy as np

DEFAULT = dict(alpha=0.3, beta=0.9, k_min=0.02, k_max=0.5, n_k=200)


def params(**overrides):
    p = dict(DEFAULT)
    p.update(overrides)
    return p


def grid(p):
    return np.linspace(p["k_min"], p["k_max"], p["n_k"])


def utility_matrix(k, alpha):
    """U[i, j] = ln(k_i^alpha - k_j), or -inf where that consumption is not positive.

    Computed once. The Bellman operator then costs one max per iteration instead of
    recomputing utility n_k * n_k times per iteration.
    """
    c = k[:, None] ** alpha - k[None, :]
    U = np.full(c.shape, -np.inf)
    U[c > 0] = np.log(c[c > 0])
    return U


def bellman(V, U, beta):
    """One application of the Bellman operator T. Returns (TV, greedy policy indices)."""
    value = U + beta * V[None, :]
    return value.max(axis=1), value.argmax(axis=1)


def vfi(p, tol=1e-6, max_iter=10000, V0=None, keep_history=False):
    """Value function iteration: iterate T from V0 until the sup-norm change is below tol."""
    k = grid(p)
    U = utility_matrix(k, p["alpha"])
    V = np.zeros(len(k)) if V0 is None else np.array(V0, dtype=float)
    dists, history = [], [V.copy()]
    for it in range(max_iter):                          # the loop
        V_new, policy = bellman(V, U, p["beta"])        # the update
        dist = np.max(np.abs(V_new - V))
        dists.append(dist)
        V = V_new                                       # a new array: no aliasing
        if keep_history:
            history.append(V.copy())
        if dist < tol:                                  # the stopping rule
            return dict(k=k, V=V, policy=policy, dists=np.array(dists), n_iter=it + 1,
                        history=history if keep_history else None)
    raise RuntimeError(f"vfi: no convergence after {max_iter} iterations")


def policy_evaluation(policy, U, beta):
    """Value of following a fixed policy forever: solve (I - beta P_sigma) v = r_sigma."""
    n = len(policy)
    r = U[np.arange(n), policy]
    P = np.zeros((n, n))
    P[np.arange(n), policy] = 1.0
    return np.linalg.solve(np.eye(n) - beta * P, r)


def howard(p, max_iter=100, policy0=None):
    """Howard's improvement (policy iteration): evaluate the policy exactly, then improve; stop when it stops changing."""
    k = grid(p)
    U = utility_matrix(k, p["alpha"])
    policy = np.zeros(len(k), dtype=int) if policy0 is None else np.array(policy0)
    policies = [policy.copy()]
    for it in range(max_iter):
        v = policy_evaluation(policy, U, p["beta"])
        _, new_policy = bellman(v, U, p["beta"])
        policies.append(new_policy.copy())
        if np.array_equal(new_policy, policy):
            return dict(k=k, V=v, policy=policy, n_iter=it + 1, policies=policies)
        policy = new_policy
    raise RuntimeError(f"howard: no convergence after {max_iter} iterations")


def closed_form_value(k, alpha, beta):
    B = alpha / (1 - alpha * beta)
    A = (1 / (1 - beta)) * ((alpha * beta / (1 - alpha * beta)) * np.log(alpha * beta) + np.log(1 - alpha * beta))
    return A + B * np.log(k)


def closed_form_policy(k, alpha, beta):
    return alpha * beta * k ** alpha


def iterations_needed(beta, tol, dist0):
    """Iterations for a contraction of modulus beta to shrink an initial error dist0 below tol."""
    return int(np.ceil(np.log(tol / dist0) / np.log(beta)))


def vfi_naive(p, tol=1e-6, max_iter=10000):
    """The same algorithm with explicit Python loops and utility recomputed every iteration. For timing only."""
    k = grid(p)
    n, alpha, beta = len(k), p["alpha"], p["beta"]
    V = np.zeros(n)
    for it in range(max_iter):
        V_new = np.empty(n)
        for i in range(n):
            best = -np.inf
            for j in range(n):
                c = k[i] ** alpha - k[j]
                if c > 0:
                    val = np.log(c) + beta * V[j]
                    if val > best:
                        best = val
            V_new[i] = best
        dist = np.max(np.abs(V_new - V))
        V = V_new
        if dist < tol:
            return V, it + 1
    raise RuntimeError("no convergence")


def timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
