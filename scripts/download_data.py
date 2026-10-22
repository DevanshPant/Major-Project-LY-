"""Download the Kaggle resume dataset via kagglehub.

Requires:  pip install kagglehub[pandas-datasets]
And a working ~/.kaggle/kaggle.json  (see kaggle docs).

Copies the CSV into data/Resume.csv so build_index.py can pick it up.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import DATASET_CSV


def main():
    try:
        import kagglehub
    except ImportError:
        print("Install first:  pip install kagglehub[pandas-datasets]")
        sys.exit(1)

    print("[download] Fetching snehaanbhawal/resume-dataset …")
    path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
    src = Path(path)
    # find the CSV
    csv_files = list(src.rglob("*.csv"))
    if not csv_files:
        print(f"[download] no CSV found under {src}")
        sys.exit(1)
    chosen = max(csv_files, key=lambda p: p.stat().st_size)
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen, DATASET_CSV)
    print(f"[download] Copied {chosen.name} → {DATASET_CSV}")


if __name__ == "__main__":
    main()
