"""Makes ``python -m utopia.baselines`` run :func:`utopia.baselines.cli.main`."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
