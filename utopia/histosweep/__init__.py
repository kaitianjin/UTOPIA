"""HistoSweep tissue masking, reduced to the code path UTOPIA uses.

Per-superpixel colour statistics feed a 2-D density filter, a GLCM texture filter refines it, and
an Otsu threshold on the standard-deviation/mean ratio contributes a second background mask. The
masks equal those of the reference HistoSweep implementation; only the texture step is batched.
"""

from .mask import tissue_mask

__all__ = ["tissue_mask"]
