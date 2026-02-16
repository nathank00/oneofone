# tests/conftest.py
"""
Shared fixtures and module loading for the test suite.

Both NBA and MLB pipelines have identically-named modules (gamelogs, train,
predict). To avoid sys.modules collisions when running the full test suite,
we load each pipeline's modules from their exact file paths using importlib
and store them under unique names.

All tests use synthetic data — no network calls, no Supabase reads/writes.
"""

import os
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure env vars exist so modules don't crash on import
os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "fake-key-for-testing")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_module_from_path(name: str, file_path: Path):
    """Load a Python module from a specific file path under a given name.
    Mocks supabase.create_client so modules don't try to connect."""
    with patch("supabase.create_client", return_value=MagicMock()):
        spec = importlib.util.spec_from_file_location(name, str(file_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Load NBA pipeline modules (unique names to avoid collisions)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT / "nba-pipeline" / "src"))

nba_gamelogs = _load_module_from_path(
    "nba_gamelogs", REPO_ROOT / "nba-pipeline" / "src" / "gamelogs.py"
)

# train.py and predict.py do `from gamelogs import fetch_paginated, supabase`
# so "gamelogs" must resolve to the NBA version while they load.
_saved = sys.modules.get("gamelogs")
sys.modules["gamelogs"] = nba_gamelogs

nba_train = _load_module_from_path(
    "nba_train", REPO_ROOT / "nba-pipeline" / "src" / "train.py"
)
nba_predict = _load_module_from_path(
    "nba_predict", REPO_ROOT / "nba-pipeline" / "src" / "predict.py"
)

if _saved is not None:
    sys.modules["gamelogs"] = _saved
else:
    sys.modules.pop("gamelogs", None)

# ---------------------------------------------------------------------------
# Load MLB pipeline modules (unique names to avoid collisions)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT / "mlb-pipeline" / "src"))

mlb_gamelogs = _load_module_from_path(
    "mlb_gamelogs", REPO_ROOT / "mlb-pipeline" / "src" / "gamelogs.py"
)

_saved = sys.modules.get("gamelogs")
sys.modules["gamelogs"] = mlb_gamelogs

mlb_train = _load_module_from_path(
    "mlb_train", REPO_ROOT / "mlb-pipeline" / "src" / "train.py"
)
mlb_predict = _load_module_from_path(
    "mlb_predict", REPO_ROOT / "mlb-pipeline" / "src" / "predict.py"
)

if _saved is not None:
    sys.modules["gamelogs"] = _saved
else:
    sys.modules.pop("gamelogs", None)
