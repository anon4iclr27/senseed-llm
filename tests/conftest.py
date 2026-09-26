"""Make the repository importable from the tests without installing it.

The package lives at the repository root; the two standalone tools live under
``scripts/`` and are deliberately *not* part of the package (they run on the
machine holding a checkpoint, where nothing is installed).  The sensitivity
tests import one of them directly in order to pin its duplicated estimator
against the package's, so ``scripts/`` goes on the path too.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
