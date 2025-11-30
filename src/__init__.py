from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXTERNAL_PATH = _PROJECT_ROOT / "external"
_external_str = str(_EXTERNAL_PATH)
if _EXTERNAL_PATH.exists() and _external_str not in sys.path:
    sys.path.insert(0, _external_str)
