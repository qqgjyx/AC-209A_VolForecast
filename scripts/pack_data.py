"""Pack the caches under data/ (CRSP rows excluded) into out/volforecast-data.tar.gz.

Usage:
    uv run python scripts/pack_data.py

Then replace the Drive file (Manage versions > Upload new version) so the notebook's link holds.
"""

from pathlib import Path

from volforecast.data import pack_snapshot, size_mb

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    out = ROOT / "out" / "volforecast-data.tar.gz"
    pack_snapshot(ROOT / "data", out)
    print(f"wrote {out} ({size_mb(out):.0f} MB)")
