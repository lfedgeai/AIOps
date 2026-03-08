#!/usr/bin/env python3
"""Run pipeline notebooks in sequence for local testing."""
import json
import os
import sys
from pathlib import Path

ELYTRA_DIR = Path(__file__).resolve().parent
os.chdir(ELYTRA_DIR)

def run_notebook(name: str, ns: dict) -> bool:
    """Execute code cells from a notebook in shared namespace. Returns True on success."""
    nb_path = ELYTRA_DIR / name
    if not nb_path.exists():
        print(f"[SKIP] {name} not found")
        return True
    with open(nb_path) as f:
        nb = json.load(f)
    print(f"\n--- Running {name} ---")
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        try:
            exec(src, ns)
        except Exception as e:
            print(f"  Cell {i} failed: {e}")
            return False
    print(f"  OK: {name}")
    return True

def main():
    # Core pipeline (00-02): runs locally with repo datasets
    core_notebooks = [
        "00_prepare_features.ipynb",
        "01_train_isolation_forest.ipynb",
        "02_test_model.ipynb",
    ]
    # Set env for pipeline paths
    os.environ.setdefault("OTEL_DATASET_DIR", str(ELYTRA_DIR.parent / "datasets"))
    os.environ.setdefault("OUT_FEATURES_CSV", str(ELYTRA_DIR / "artifacts/features.csv"))
    os.environ.setdefault("TRAIN_DATA_PATH", str(ELYTRA_DIR / "artifacts/features.csv"))
    os.environ.setdefault("TEST_DATA_PATH", str(ELYTRA_DIR / "artifacts/features.csv"))
    os.environ.setdefault("MODEL_OUTPUT_PATH", str(ELYTRA_DIR / "artifacts/model.pkl"))
    os.environ.setdefault("MODEL_PATH", str(ELYTRA_DIR / "artifacts/model.pkl"))
    os.environ.setdefault("FEATURE_COLS_PATH", str(ELYTRA_DIR / "artifacts/feature_cols.json"))
    os.environ.setdefault("THRESHOLD_PATH", str(ELYTRA_DIR / "artifacts/threshold.json"))
    os.environ.setdefault("TEST_REPORT_PATH", str(ELYTRA_DIR / "artifacts/test_report.json"))

    ns = {"__name__": "__main__"}
    for nb in core_notebooks:
        if not run_notebook(nb, ns):
            print(f"\nFAILED: {nb}")
            return 1

    print("\n--- Core pipeline (00-02) OK ---")

    # 03/04: require AWS and OpenShift - verify they fail gracefully
    for nb, skip_reason in [
        ("03_copy_to_s3.ipynb", "AWS credentials"),
        ("04_deploy_to_serving.ipynb", "oc/kubectl + cluster"),
    ]:
        try:
            if run_notebook(nb, ns):
                print(f"  {nb}: ran (credentials available)")
            else:
                print(f"  {nb}: failed (expected without {skip_reason})")
        except Exception as e:
            print(f"  {nb}: {type(e).__name__} (expected without {skip_reason})")

    print("\n--- Pipeline test complete ---")
    return 0

if __name__ == "__main__":
    sys.exit(main())
