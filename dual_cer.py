import numpy as np

from Simplex_Algorithm import LinearProgram


def dual_certificate(lp: LinearProgram, x: np.ndarray, lam: np.ndarray, eps: float = 1e-6) -> bool:
    """Check primal/dual feasibility and strong duality certificate."""
    A = lp.A
    b = lp.b
    c = lp.c

    primal_feasible = np.all(x >= -eps) and np.allclose(A @ x, b, atol=eps)
    dual_feasible = np.all((A.T @ lam) <= c + eps)
    complementary = np.isclose(c @ x, b @ lam, atol=eps)

    return bool(primal_feasible and dual_feasible and complementary)
