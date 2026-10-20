"""Make the stage folders under ``scripts/`` importable from the tests (``import train_series_classifier``)."""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for folder in ("data", "cascade", "networks", "meta", "rotation"):
    path = str(_SCRIPTS / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
