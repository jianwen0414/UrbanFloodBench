"""
Configuration — paths, constants, and hyperparameter defaults.

Change RAW_DATA_PATH to point at wherever the raw competition data is
mounted / downloaded.  On Google Colab with Drive mounted the default
works out-of-the-box; locally you can override it via the environment
variable ``FLOOD_DATA_PATH``.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

# Default assumes Google Drive is mounted at /content/drive on Colab.
# Override by setting the FLOOD_DATA_PATH environment variable.
RAW_DATA_PATH: Path = Path(
    os.environ.get("FLOOD_DATA_PATH", "/content/drive/MyDrive/UrbanFloodProject")
)

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
