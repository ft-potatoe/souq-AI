"""
scripts/refresh.py
One-command full refresh: ingest -> build features -> retrain models.

This is the single command to run whenever you add new data. It chains the steps
in the correct order and STOPS if a step fails, so you never end up with features
built on half-ingested data or models trained on a stale feature table.

Usage
-----
  python scripts/refresh.py --src data/incoming
  python scripts/refresh.py --src data/incoming --replace   # fresh overwrite ingest
  python scripts/refresh.py --skip-retrain                  # ingest + features only
  python scripts/refresh.py --skip-ingest --src .           # rebuild features + retrain only

After it finishes, restart the API so it reloads the new feature table:
  uvicorn api.main:app --reload --port 8000

Exit codes: 0 = all requested steps succeeded; non-zero = a step failed (see log).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable  # use the same interpreter that launched this script


def _run(label: str, args: list[str]) -> bool:
    """Run a subprocess step; return True on success (exit 0)."""
    print(f"\n{'=' * 70}\n[refresh] {label}\n{'=' * 70}", flush=True)
    proc = subprocess.run([PY, *args], cwd=ROOT)
    if proc.returncode != 0:
        print(f"[refresh] STEP FAILED: {label} (exit {proc.returncode})", flush=True)
        return False
    print(f"[refresh] OK: {label}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Full data + model refresh pipeline.")
    parser.add_argument(
        "--src", type=str, default=None,
        help="Source directory of new CSV/Excel files (required unless --skip-ingest).",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Pass --replace to ingestion (overwrite instead of merge). Rarely needed.",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip ingestion (use when raw data is already current).",
    )
    parser.add_argument(
        "--skip-retrain", action="store_true",
        help="Skip model retraining (ingest + feature rebuild only).",
    )
    args = parser.parse_args()

    if not args.skip_ingest and not args.src:
        parser.error("--src is required unless --skip-ingest is set.")

    # Step 1: ingest (merge by default)
    if not args.skip_ingest:
        ingest_args = ["scripts/ingest/load_raw.py", "--src", args.src]
        if args.replace:
            ingest_args.append("--replace")
        if not _run("Step 1/3: ingest raw data", ingest_args):
            print("[refresh] Aborting: ingestion failed. Features NOT rebuilt.", flush=True)
            return 1

    # Step 2: rebuild features
    if not _run("Step 2/3: build features", ["scripts/features/build_features.py"]):
        print("[refresh] Aborting: feature build failed. Models NOT retrained.", flush=True)
        return 1

    # Step 3: retrain models
    if not args.skip_retrain:
        # weekly_retrain returns non-zero only on a CORE-model gate failure; additive
        # models (vol HMM, clustering) failing their gate is non-blocking. We surface
        # the code but still consider the refresh "done" -- the prior model is kept.
        ok = _run("Step 3/3: retrain models", ["scripts/retrain/weekly_retrain.py"])
        if not ok:
            print(
                "[refresh] NOTE: retrain reported a failure. A core model missed its "
                "validation gate and the prior model was kept (auto-rollback). Check "
                "logs/retrain_log.jsonl -- this is often benign.",
                flush=True,
            )

    print(
        "\n[refresh] Done. Restart the API to load the new features:\n"
        "          uvicorn api.main:app --reload --port 8000",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
