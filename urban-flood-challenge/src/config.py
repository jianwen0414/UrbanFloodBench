"""
Configuration — paths, constants, and hyperparameter defaults.

Change RAW_DATA_PATH to point at wherever the raw competition data is
mounted / downloaded.  On Google Colab with Drive mounted the default
works out-of-the-box; locally you can override it via the environment
variable ``FLOOD_DATA_PATH`` (set in ``src/.env``).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env (if python-dotenv is installed)
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).resolve().parent / ".env"

try:
    from dotenv import load_dotenv

    if _ENV_FILE.is_file():
        load_dotenv(_ENV_FILE, override=False)
except ModuleNotFoundError:
    # python-dotenv is not installed — rely on real env vars.
    pass

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

# Default assumes Google Drive is mounted at /content/drive on Colab.
# Override by setting the FLOOD_DATA_PATH environment variable.
_raw = os.environ.get("FLOOD_DATA_PATH", "/content/drive/MyDrive/UrbanFloodProject")

# Guard: reject URLs that were accidentally pasted as paths
if _raw.startswith(("http://", "https://")):
    warnings.warn(
        f"FLOOD_DATA_PATH looks like a URL, not a filesystem path: {_raw!r}. "
        "Please set it to a local directory (see src/.env for examples)."
    )

RAW_DATA_PATH: Path = Path(_raw)

# Convenience sub-paths (adjust if the Drive folder layout changes)
MODELS_DIR: Path = RAW_DATA_PATH / "Models"

# ---------------------------------------------------------------------------
# Processed / cached artefacts
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CACHE_DIR: Path = PROJECT_ROOT / "data" / ".cache"

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
