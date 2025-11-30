#!/usr/bin/env python
"""
Launch the FastAPI inference server.

Example:
    python scripts/serve_api.py --checkpoint checkpoints/humbugdb_baseline/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from mosquito_monitor.api import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve mosquito classifier API.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to trained model checkpoint (.pt) produced by scripts/train_model.py",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(checkpoint_path=args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
