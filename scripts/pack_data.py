"""Pack the caches under data/ (CRSP excluded) into one archive for the shared snapshot.

Usage:
    uv run python scripts/pack_data.py            # writes out/volforecast-data.tar.gz

Replace the file on Google Drive with the new archive (Manage versions > Upload new version), so
the link in ms2.ipynb's setup cell keeps working.
"""

import argparse
from pathlib import Path

from volforecast.data import pack_snapshot, size_mb

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "volforecast-data.tar.gz")
    args = parser.parse_args()
    pack_snapshot(ROOT / "data", args.out)
    print(f"wrote {args.out} ({size_mb(args.out):.0f} MB)")


if __name__ == "__main__":
    main()
