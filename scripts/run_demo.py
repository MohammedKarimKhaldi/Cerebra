"""End-to-end demo: generates synthetic data and runs the full triage pipeline.

Usage:
    python scripts/run_demo.py [--output-dir /tmp/cerebra_demo]

Success criterion: completes in under 60 seconds on CPU and prints a valid JSON report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/cerebra_demo"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: generate synthetic study
    fixture_dir = REPO_ROOT / "tests" / "fixtures" / "synthetic"
    print(f"[demo] Generating synthetic BraTS fixture → {fixture_dir}")
    from make_synthetic_sample import generate  # type: ignore[import]

    generate(fixture_dir, seed=42)

    # Step 2: run the triage pipeline
    from cerebra.pipeline import run_triage

    print(f"[demo] Running triage pipeline…")
    t0 = time.monotonic()
    report = run_triage(
        input_path=fixture_dir,
        output_dir=output_dir,
        study_id="demo01",
    )
    elapsed = time.monotonic() - t0

    # Step 3: print report
    print("\n" + "=" * 60)
    print("CEREBRA TRIAGE REPORT")
    print("=" * 60)
    print(json.dumps(report.model_dump(), indent=2))
    print("=" * 60)
    print(f"\nCompleted in {elapsed:.1f}s")

    if elapsed > 60:
        print(f"WARNING: demo took {elapsed:.1f}s (target < 60s)")
    else:
        print("✓ Within 60s target")

    if report.visualization_path:
        png = Path(report.visualization_path)
        if png.exists():
            print(f"✓ Overlay PNG: {png}  ({png.stat().st_size / 1024:.1f} kB)")
        else:
            print(f"✗ Overlay PNG not found at {png}")


if __name__ == "__main__":
    main()
