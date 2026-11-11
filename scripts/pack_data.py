"""Pack the team bucket: every cache under data/ (CRSP included) and a volforecast wheel.

Usage:
    uv run python scripts/pack_data.py   # writes out/volforecast-bucket.tar.gz

The bucket is for the team and the TF only (CRSP under the team's WRDS licence), so its link
lives only in this private repository. Rerun it whenever data/ or src/ changes, since a standalone
notebook runs the wheel, and replace the Drive file with Manage versions > Upload new
version, so the link in ms2.ipynb holds.
"""

import subprocess
import tarfile
import tempfile
from pathlib import Path

from volforecast.data import size_mb

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"README.md", ".gitkeep", "rv5_build.log"}  # tracked files and local logs

if __name__ == "__main__":
    out = ROOT / "out" / "volforecast-bucket.tar.gz"
    with tempfile.TemporaryDirectory() as wheels, tarfile.open(out, "w:gz") as tar:
        subprocess.run(["uv", "build", "--wheel", "-q", "-o", wheels], cwd=ROOT, check=True)
        for p in sorted((ROOT / "data").iterdir()):
            if p.name not in SKIP:
                tar.add(p, arcname=f"data/{p.name}")
        for wheel in Path(wheels).glob("*.whl"):
            tar.add(wheel, arcname=f"dist/{wheel.name}")
    print(f"wrote {out} ({size_mb(out):.0f} MB)")
