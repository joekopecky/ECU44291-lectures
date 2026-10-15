"""The three-size cake on a grid: value function iteration and policy iteration by hand-sized arrays.

Lecture 5 of ECU44291. Cake sizes w in {0, 1, 2}, keep w' <= w, u(c) = sqrt(c), beta = 0.9.
Small enough to do on the board; the code is here so the board numbers can be checked.
"""
import numpy as np


def utility_matrix(sizes=(0, 1, 2)):
    """U[i, j] = sqrt(w_i - w_j) for w_j <= w_i, -inf otherwise."""
    w = np.asarray(sizes, dtype=float)
    U = np.full((len(w), len(w)), -np.inf)
    for i in range(len(w)):
        for j in range(i + 1):
            U[i, j] = np.sqrt(w[i] - w[j])
    return U


def value_iteration(beta=0.9, n_iter=3, sizes=(0, 1, 2)):
    """Returns the list of (V_n, policy_n) for n = 1..n_iter from V_0 = 0."""
    U = utility_matrix(sizes)
    V = np.zeros(len(sizes))
    out = []
    for _ in range(n_iter):
        value = U + beta * V[None, :]
        V, policy = value.max(axis=1), value.argmax(axis=1)
        out.append((V.copy(), policy.copy()))
    return out


def policy_evaluation(policy, U, beta):
    """The value of following a fixed policy forever: solve (I - beta P) v = r."""
    n = len(policy)
    r = U[np.arange(n), policy]
    P = np.zeros((n, n))
    P[np.arange(n), policy] = 1.0
    return np.linalg.solve(np.eye(n) - beta * P, r), r, P


def policy_iteration(beta=0.9, sizes=(0, 1, 2), max_iter=20):
    """Howard's improvement from 'keep 0 everywhere'. Returns the list of (policy, r, v, improved policy) per step."""
    U = utility_matrix(sizes)
    policy = np.zeros(len(sizes), dtype=int)
    steps = []
    for _ in range(max_iter):
        v, r, _ = policy_evaluation(policy, U, beta)
        new_policy = (U + beta * v[None, :]).argmax(axis=1)
        steps.append((policy.copy(), r, v, new_policy.copy()))
        if np.array_equal(new_policy, policy):
            return steps
        policy = new_policy
    raise RuntimeError("policy iteration did not stop")
