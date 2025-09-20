from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import yaml
from sklearn.linear_model import LinearRegression

from .data import load_data_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Linear Regression model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
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

    data_path = cfg.get("data_path", "datasets/student_scores_dataset.csv")
    test_size = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 42))
    target = cfg.get("target", "score")
    feature = cfg.get("feature")
    model_path = cfg.get("model_path", "models/model.joblib")

    feature_columns = None
    if feature is not None:
        # allow single name or list in config; keep as-is
        feature_columns = feature if isinstance(feature, list) else [feature]

    splits = load_data_pipeline(
        data_path=data_path,
        feature_columns=feature_columns,  # defaults inside data.py to [hours_studied, exam_difficulty]
        target_column=target,
        test_size=test_size,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(splits.X_train, splits.y_train)

    # Attach feature names for consistent inference later
    artifact = {
        "model": model,
        "feature_names": list(splits.X_train.columns),
    }

    model_out = Path(model_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)
    print(f"Saved model to {model_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


