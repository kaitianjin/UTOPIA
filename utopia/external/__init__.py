"""Front end for predictions made by another model (``mode: external``).

The user gives one folder per sample holding the other model's predictions, the measured values
of the calibration samples and the position of every spot on the raw H&E. ``import`` checks the
folders, writes the calibration and prediction stores and masks the tissue of each H&E;
``features`` encodes a crop of every spot with UNI; ``cluster`` clusters the spot features and
writes the labels into both stores. The last two do nothing when the user supplies the labels.
"""

#: Stage order of an external run that lists sample folders.
STAGES = ["import", "features", "cluster", "infer"]

__all__ = ["STAGES"]
