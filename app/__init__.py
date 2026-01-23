from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

# Allow `python -m app.cli` from repo root by pointing this package to src/app.
__path__ = list(extend_path(__path__, __name__))
_src_app = Path(__file__).resolve().parents[1] / "src" / "app"
if _src_app.exists():
    __path__.append(str(_src_app))
