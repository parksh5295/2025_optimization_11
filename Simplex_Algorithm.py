import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass
class LinearProgram:
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray


def get_vertex(B: Iterable[int], lp: LinearProgram, *, one_based: bool = True) -> np.ndarray:
    """Construct a basic feasible solution from a basis index set."""
    b_inds = np.array(sorted(B), dtype=int)

    if one_based:
        b_inds = b_inds - 1

    AB = lp.A[:, b_inds]
    xB = np.linalg.solve(AB, lp.b)

    x = np.zeros(lp.c.shape, dtype=float)
    x[b_inds] = xB
    return x


def edge_transition(lp: LinearProgram, B: Iterable[int], q: int, *, one_based: bool = True) -> tuple[int, float]:
    """Perform the edge transition pivot test used in the simplex method."""
    A = lp.A
    b = lp.b

    n = A.shape[1]
    b_inds = np.array(sorted(B), dtype=int)

    if one_based:
        b_inds = b_inds - 1
        q = q - 1

    all_indices = np.arange(n, dtype=int)
    mask = np.ones(n, dtype=bool)
    mask[b_inds] = False
    n_inds = all_indices[mask]

    AB = A[:, b_inds]
    direction = np.linalg.solve(AB, A[:, n_inds[q]])
    xB = np.linalg.solve(AB, b)

    pivot_index = -1 if not one_based else 0
    min_ratio = np.inf

    for idx, d_i in enumerate(direction):
        if d_i > 0:
            ratio = xB[idx] / d_i
            if ratio < min_ratio:
                pivot_index = idx + 1 if one_based else idx
                min_ratio = ratio

    return pivot_index, float(min_ratio)


def step_lp(B: Iterable[int], lp: LinearProgram, *, one_based: bool = True) -> Tuple[list[int], bool]:
    """Single simplex iteration returning the updated basis and optimality flag."""
    A = lp.A
    b = lp.b
    c = lp.c

    n = A.shape[1]
    basis_list = list(B)

    sorted_basis_zero = np.array(sorted(basis_list), dtype=int)
    if one_based:
        sorted_basis_zero = sorted_basis_zero - 1

    all_indices = np.arange(n, dtype=int)
    mask = np.ones(n, dtype=bool)
    mask[sorted_basis_zero] = False
    non_basis_zero = all_indices[mask]

    AB = A[:, sorted_basis_zero]
    AV = A[:, non_basis_zero]

    xB = np.linalg.solve(AB, b)
    cB = c[sorted_basis_zero]
    lam = np.linalg.solve(AB.T, cB)
    cV = c[non_basis_zero]
    muV = cV - AV.T @ lam

    chosen_q = None
    pivot_row_index = None
    entering_value = np.inf
    best_delta = np.inf

    for idx, mu_i in enumerate(muV):
        if mu_i < 0:
            pi, xi_prime = edge_transition(lp, basis_list, idx + 1 if one_based else idx, one_based=one_based)
            product = mu_i * xi_prime
            if product < best_delta:
                chosen_q = idx
                pivot_row_index = pi
                entering_value = xi_prime
                best_delta = product

    if chosen_q is None:
        return basis_list, True

    if np.isinf(entering_value):
        raise RuntimeError("unbounded")

    sorted_basis_original = np.array(sorted(basis_list), dtype=int)
    if one_based:
        pivot_basis_value = sorted_basis_original[pivot_row_index - 1]
        entering_index_value = non_basis_zero[chosen_q] + 1
    else:
        pivot_basis_value = sorted_basis_original[pivot_row_index]
        entering_index_value = int(non_basis_zero[chosen_q])

    pivot_position = basis_list.index(int(pivot_basis_value))
    basis_list[pivot_position] = int(entering_index_value)

    return basis_list, False


def minimize_lp_basis(B: Iterable[int], lp: LinearProgram, *, one_based: bool = True) -> list[int]:
    """Iterate simplex steps until optimality for the provided basis."""
    basis = list(B)
    done = False
    while not done:
        basis, done = step_lp(basis, lp, one_based=one_based)
    return basis


def minimize_lp(lp: LinearProgram, *, one_based: bool = True) -> np.ndarray:
    """Two-phase simplex to obtain an optimal basic feasible solution."""
    A = lp.A
    b = lp.b
    c = lp.c
    m, n = A.shape

    z = np.ones(m, dtype=float)
    diag_entries = np.where(b >= 0, 1.0, -1.0)
    Z = np.diag(diag_entries)

    A_prime = np.hstack([A, Z])
    b_prime = b.copy()
    c_prime = np.concatenate([np.zeros(n, dtype=float), z])
    lp_init = LinearProgram(A_prime, b_prime, c_prime)

    if one_based:
        basis = list(range(n + 1, n + m + 1))
    else:
        basis = list(range(n, n + m))

    basis = minimize_lp_basis(basis, lp_init, one_based=one_based)

    if one_based:
        infeasible = any(idx > n for idx in basis)
    else:
        infeasible = any(idx >= n for idx in basis)
    if infeasible:
        raise RuntimeError("infeasible")

    top_block = np.hstack([A, np.eye(m)])
    bottom_block = np.hstack([np.zeros((m, n)), np.eye(m)])
    A_opt = np.vstack([top_block, bottom_block])
    b_opt = np.concatenate([b, np.zeros(m, dtype=float)])
    c_opt = np.concatenate([c, np.zeros(m, dtype=float)])

    lp_opt = LinearProgram(A_opt, b_opt, c_opt)
    basis = minimize_lp_basis(basis, lp_opt, one_based=one_based)

    vertex = get_vertex(basis, lp_opt, one_based=one_based)
    return vertex[:n]
