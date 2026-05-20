#!/usr/bin/env python3
"""Download Office-31 images to a path **outside** the git repo.

Does not commit data. After download, run:

  python examples/21_benchmark_sklearn_table.py --office31-root YOUR_ROOT
  python scripts/generate_reference_benchmark.py --office31-root YOUR_ROOT

See docs/walkthroughs/19-office31-real-data.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Office-31 (amazon, dslr, webcam)")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Target directory (e.g. D:/datasets/office31). Created if missing.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Override download URL (default: Georgia Tech office31.tar)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if domains already exist",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check layout under --root; do not download",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    from pmh.datasets.office31 import download_office31, list_office31_domains, verify_office31_layout

    if args.verify_only:
        verify_office31_layout(root)
        print(f"OK: Office-31 layout under {root} ({', '.join(list_office31_domains())})")
        return

    archive = download_office31(root, url=args.url, force=args.force)
    verify_office31_layout(root)
    print(f"Done. Archive: {archive}")
    print(f"Domains: {', '.join(list_office31_domains())} under {root}")
    print("\nNext:")
    print(f"  python examples/21_benchmark_sklearn_table.py --office31-root {root}")


if __name__ == "__main__":
    main()
