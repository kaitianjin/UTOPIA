"""Harmony batch-effect correction of patch-level embeddings.

Slides differ in staining, scanner and fixation, which a model trained on one slide and applied to
another would read as biology. Out-of-sample runs therefore correct the window-level global
(class-token) embeddings and re-fuse them; in-sample runs and the local stream are left alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def correct(
    embeddings: np.ndarray,
    batches: Sequence[str],
    max_iter: int = 20,
    random_state: int = 0,
) -> np.ndarray:
    """Regress the batch signal out of ``(N, D)`` embeddings, given one batch label per row.

    Returns the corrected embeddings as float32, row for row.
    """
    import harmonypy
    import pandas as pd

    if len(batches) != len(embeddings):
        raise ValueError(f"{len(embeddings)} embeddings but {len(batches)} batch labels")
    result = harmonypy.run_harmony(
        np.ascontiguousarray(embeddings, dtype=np.float64),
        pd.DataFrame({"batch": list(batches)}),
        ["batch"],
        max_iter_harmony=max_iter,
        random_state=random_state,
    )
    return np.asarray(result.Z_corr).T.astype(np.float32)
