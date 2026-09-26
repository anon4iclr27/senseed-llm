"""Put the repository root on ``sys.path``.

Every script in this directory is meant to be runnable with a bare
``python experiments/foo.py`` from a fresh clone, without installing anything.
Importing this module first is what makes ``import senseed`` work in that case; it
is a no-op once the package is installed.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def require(path: str) -> str:
    """Fail with an explanation rather than a bare FileNotFoundError.

    Weight slices are the model owners' under their own licenses and are not
    shipped here; `scripts/extract_weights.py` regenerates them from a
    checkpoint in a few seconds.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n"
            "Weight slices are not included in this repository.  Regenerate:\n"
            "  python3 scripts/extract_weights.py <checkpoint-dir> "
            f"{path}\n"
            "and run this script from the repository root.")
    return path
