from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_data_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
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
        feature_columns = feature if isinstance(feature, list) else [feature]

    splits = load_data_pipeline(
        data_path=data_path,
        feature_columns=feature_columns,
        target_column=target,
        test_size=test_size,
        random_state=random_state,
    )

    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact

    preds = model.predict(splits.X_test)
    mae = mean_absolute_error(splits.y_test, preds)
    mse = mean_squared_error(splits.y_test, preds)
    rmse = mse ** 0.5  # Calculate RMSE manually for compatibility
    r2 = r2_score(splits.y_test, preds)

    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2:  {r2:.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)


