import numpy as np

from Simplex_Algorithm import (
    LinearProgram,
    edge_transition,
    get_vertex,
    minimize_lp,
    minimize_lp_basis,
    step_lp,
)
from dual_cer import dual_certificate


def build_demo_lp() -> LinearProgram:
    """Create a small LP for demonstration purposes.

    Problem: maximize 3 x1 + 2 x2 subject to
        x1 + x2 <= 4
        2 x1 + x2 <= 5
        x >= 0

    We rewrite it as a minimization with slack variables and equality constraints.
    """
    A = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
        ]
    )
    b = np.array([4.0, 5.0])
    c = np.array([-3.0, -2.0, 0.0, 0.0])  # minimize the negative of the objective
    return LinearProgram(A, b, c)


def example_get_vertex(lp: LinearProgram) -> None:
    basis = [3, 4]
    vertex = get_vertex(basis, lp)
    print("get_vertex with basis {3,4} ->", vertex)


def example_edge_transition(lp: LinearProgram) -> None:
    basis = [3, 4]
    pivot_index, step_size = edge_transition(lp, basis, q=1)
    print("edge_transition entering variable q=1 -> pivot row", pivot_index, "step", step_size)


def example_step_lp(lp: LinearProgram) -> None:
    basis = [3, 4]
    basis, optimal = step_lp(basis, lp)
    print("step_lp first iteration -> basis", basis, "optimal?", optimal)
    if not optimal:
        basis, optimal = step_lp(basis, lp)
        print("step_lp second iteration -> basis", basis, "optimal?", optimal)


def example_minimize_lp_basis(lp: LinearProgram) -> list[int]:
    basis = [3, 4]
    final_basis = minimize_lp_basis(basis, lp)
    print("minimize_lp_basis starting from slack basis ->", final_basis)
    return final_basis


def example_minimize_lp(lp: LinearProgram) -> np.ndarray:
    solution = minimize_lp(lp)
    print("minimize_lp two-phase solution ->", solution)
    return solution


def example_dual_certificate(lp: LinearProgram) -> None:
    basis = minimize_lp_basis([3, 4], lp)
    x = get_vertex(basis, lp)

    basis_zero = np.array(sorted(basis), dtype=int) - 1
    AB = lp.A[:, basis_zero]
    cB = lp.c[basis_zero]
    lam = np.linalg.solve(AB.T, cB)

    certified = dual_certificate(lp, x, lam)
    print("dual_certificate with optimal x and lambda ->", certified)
    print("lambda ->", lam)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    lp = build_demo_lp()

    example_get_vertex(lp)
    example_edge_transition(lp)
    example_step_lp(lp)

    final_basis = example_minimize_lp_basis(lp)
    solution = example_minimize_lp(lp)

    print("final basis from minimize_lp_basis ->", final_basis)
    print("objective value c^T x ->", lp.c @ get_vertex(final_basis, lp))
    print("solution from minimize_lp ->", solution)

    example_dual_certificate(lp)


if __name__ == "__main__":
    main()
