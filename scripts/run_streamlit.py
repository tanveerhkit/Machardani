#!/usr/bin/env python
"""
Run the Streamlit front-end for mosquito classification.

Example:
    python scripts/run_streamlit.py --checkpoint checkpoints/humbugdb_baseline/best_model.pt
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Streamlit mosquito app.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to trained checkpoint file.",
    )
    parser.add_argument("--port", type=int, default=8501)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    entry_script = Path(__file__).resolve().parent.parent / "src" / "mosquito_monitor" / "app" / "streamlit_app.py"
    subprocess.run(
        [
            "streamlit",
            "run",
            str(entry_script),
            "--server.port",
            str(args.port),
            "--",
            "--checkpoint",
            str(args.checkpoint),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
