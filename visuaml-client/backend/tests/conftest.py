"""Resolve the project package path before pytest imports any test module.

Without this explicit bootstrap, collection order can accidentally determine
whether 'visuaml' is importable on a clean CI runner.
"""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
