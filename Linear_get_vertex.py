import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class LinearProgram:
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray

def get_vertex(B: Iterable[int], lp: LinearProgram, *, one_based: bool = True) -> np.ndarray:
    """Replicates the Julia get_vertex helper.

    Args:
        B: Indices of the basic columns.
        lp: LinearProgram instance.
        one_based: Set to False if B already uses 0-based indices.

    Returns:
        Basic feasible solution x that satisfies A[:, B] @ x[B] = b.
    """
    b_inds = np.array(sorted(B), dtype=int)
    if one_based:
        b_inds -= 1  # convert Julia-style indices to Python 0-based

    AB = lp.A[:, b_inds]
    xB = np.linalg.solve(AB, lp.b)

    x = np.zeros(lp.c.shape, dtype=float)
    x[b_inds] = xB
    return x