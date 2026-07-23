"""Pure search-space decisions shared by optimiser tests and runtime."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def vector_scope_choices(config, steering_vectors: Iterable[Any]) -> list[str]:
    """Return a stable vector-scope distribution for the whole study.

    Rank-k tensors have shape ``(directions, layers, hidden)``.  Global layer
    interpolation is not implemented for that layout, so a study containing
    any rank-k variant must use per-layer scope for every trial.  Resolving
    this once avoids an Optuna categorical distribution that changes between
    variants.
    """
    fixed = config.steering.fixed_vector_scope
    has_multi_direction = any(
        getattr(vectors, "ndim", None) == 3 for vectors in steering_vectors
    )
    if has_multi_direction:
        if fixed not in (None, "per layer"):
            raise ValueError(
                "fixed_vector_scope='global' is incompatible with "
                "multi-direction steering; use 'per layer'."
            )
        return ["per layer"]
    return [fixed] if fixed else ["global", "per layer"]
