from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    """Container for train/test splits."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _ensure_columns_exist(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Validate that all required columns are present; raise on missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            f". Available columns: {list(df.columns)}"
        )


def load_csv(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame with basic validation."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    return df


# No special parsing for hours; 4.37 is treated as 4.37 hours.


def prepare_features_and_target(df: pd.DataFrame,
    feature_columns: Union[str, Sequence[str], None] = None,
    target_column: str = "score",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features X and target y.

    This function validates column presence and returns copies to avoid
    accidental mutation of the original DataFrame.
    """
    # Default to a two-feature setup if not provided, per the updated dataset
    if feature_columns is None:
        feature_columns = ["hours_studied", "exam_difficulty"]

    # Allow passing a single column name as a string
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]

    required_columns: List[str] = list(feature_columns) + [target_column]
    _ensure_columns_exist(df, required_columns)

    X = df[list(feature_columns)].copy()
    y = df[target_column].copy()

    # Treat hours_studied as a numeric decimal value in hours

    # One-hot encode exam_difficulty (Easy/Medium/Hard)
    if "exam_difficulty" in X.columns:
        categories = ["Easy", "Medium", "Hard"]
        X.loc[:, "exam_difficulty"] = pd.Categorical(
            X["exam_difficulty"], categories=categories
        )
        dummies = pd.get_dummies(
            X["exam_difficulty"], prefix="difficulty", dtype=float
        )
        X = X.drop(columns=["exam_difficulty"]).join(dummies)

    # Coerce numerics and drop rows with NaNs
    numeric_cols = X.columns
    X.loc[:, numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid_mask = X.notna().all(axis=1) & y.notna()
    if not valid_mask.all():
        X = X[valid_mask]
        y = y[valid_mask]

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplits:
    """Split features and target into train/test sets.

    For regression problems we do not stratify by default.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return DatasetSplits(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def load_data_pipeline(
    data_path: Union[str, Path],
    feature_columns: Union[str, Sequence[str], None] = None,
    target_column: str = "score",
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplits:
    """Convenience function: load CSV → prepare X/y → split train/test."""
    df = load_csv(data_path)
    X, y = prepare_features_and_target(df, feature_columns, target_column)
    return split_train_test(X, y, test_size=test_size, random_state=random_state)


__all__ = [
    "DatasetSplits",
    "load_csv",
    "prepare_features_and_target",
    "split_train_test",
    "load_data_pipeline",
]


