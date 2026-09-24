"""CLI for the offline Race Score prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from race_score_model import RaceScoreModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=float, required=True)
    parser.add_argument("--elevation", type=float, required=True)
    parser.add_argument("--time", required=True)
    parser.add_argument("--anchor-time")
    parser.add_argument("--anchor-score", type=float)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "model_artifacts" / "race_score_model_v1.json",
    )
    args = parser.parse_args()

    model = RaceScoreModel.load(args.model)
    estimate = model.estimate(
        distance_km=args.distance,
        elevation_gain_m=args.elevation,
        finish_time=args.time,
        anchor_time=args.anchor_time,
        anchor_score=args.anchor_score,
    )
    print(json.dumps(estimate.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

