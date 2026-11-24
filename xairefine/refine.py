"""XAI-guided architecture refinement (Sec. 3.5).

1. Train the baseline CNN.
2. Compute m_l for every block (Eq. 4) on a reference set (the validation split).
3. tau = Percentile_p(m_l) (Eq. 5).
4. Remove the blocks with m_l < tau, rebuild the network with the remaining widths and retrain with the same protocol.
"""
from __future__ import annotations

import numpy as np


def threshold(relevance: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(relevance, dtype=float), percentile))


def blocks_to_keep(relevance: np.ndarray, percentile: float = 25.0, min_keep: int = 2) -> list[int]:
    """Indices of the blocks that survive; always keeps the ``min_keep`` most relevant blocks."""
    rel = np.asarray(relevance, dtype=float)
    tau = threshold(rel, percentile)
    keep = [i for i, r in enumerate(rel) if r >= tau]
    if len(keep) < min_keep:
        keep = sorted(np.argsort(-rel)[:min_keep].tolist())
    return keep


def refined_filters(filters: list[int], keep: list[int]) -> list[int]:
    return [filters[i] for i in keep]
