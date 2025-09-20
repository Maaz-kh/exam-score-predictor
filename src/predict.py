from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict exam score")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--hours",
        type=float,
        required=True,
        help="Hours studied as a decimal (e.g., 4.5)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["Easy", "Medium", "Hard"],
        default="Medium",
        help="Exam difficulty category",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_path = cfg.get("model_path", "models/model.joblib")

    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    feature_names = (
        artifact.get("feature_names", None) if isinstance(artifact, dict) else None
    )

    # Construct a single-row feature frame matching training features
    base = {
        "hours_studied": args.hours,
        "difficulty_Easy": 1.0 if args.difficulty == "Easy" else 0.0,
        "difficulty_Medium": 1.0 if args.difficulty == "Medium" else 0.0,
        "difficulty_Hard": 1.0 if args.difficulty == "Hard" else 0.0,
    }
    X = pd.DataFrame([base])

    if feature_names is not None:
        # Add any missing columns as zeros and order columns to match training
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_names]

    pred = float(model.predict(X)[0])
    print(f"Predicted score: {pred:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        sys.exit(1)


