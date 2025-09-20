import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_csv, prepare_features_and_target, split_train_test


def test_load_csv():
    """Test CSV loading functionality."""
    # Create a temporary CSV for testing
    test_data = {
        "hours_studied": [4.37, 9.56, 7.59],
        "exam_difficulty": ["Easy", "Medium", "Hard"],
        "score": [60.2, 100.0, 93.9]
    }
    df = pd.DataFrame(test_data)
    test_csv_path = "test_data.csv"
    df.to_csv(test_csv_path, index=False)
    
    try:
        loaded_df = load_csv(test_csv_path)
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["hours_studied", "exam_difficulty", "score"]
    finally:
        Path(test_csv_path).unlink(missing_ok=True)


def test_prepare_features_and_target():
    """Test feature and target preparation."""
    test_data = {
        "hours_studied": [4.37, 9.56, 7.59],
        "exam_difficulty": ["Easy", "Medium", "Hard"],
        "score": [60.2, 100.0, 93.9]
    }
    df = pd.DataFrame(test_data)
    
    X, y = prepare_features_and_target(df)
    
    # Check that hours_studied is preserved as numeric
    assert "hours_studied" in X.columns
    assert X["hours_studied"].dtype in ["float64", "int64"]
    
    # Check that exam_difficulty is one-hot encoded
    assert "difficulty_Easy" in X.columns
    assert "difficulty_Medium" in X.columns
    assert "difficulty_Hard" in X.columns
    assert "exam_difficulty" not in X.columns
    
    # Check target
    assert len(y) == 3
    assert y.dtype in ["float64", "int64"]


def test_split_train_test():
    """Test train/test splitting."""
    test_data = {
        "hours_studied": [4.37, 9.56, 7.59, 6.39, 2.4],
        "exam_difficulty": ["Easy", "Medium", "Hard", "Easy", "Medium"],
        "score": [60.2, 100.0, 93.9, 76.6, 45.5]
    }
    df = pd.DataFrame(test_data)
    
    X, y = prepare_features_and_target(df)
    splits = split_train_test(X, y, test_size=0.2, random_state=42)
    
    # Check that splits have correct sizes
    assert len(splits.X_train) + len(splits.X_test) == len(X)
    assert len(splits.y_train) + len(splits.y_test) == len(y)
    
    # Check that test size is approximately correct
    expected_test_size = len(X) * 0.2
    assert abs(len(splits.X_test) - expected_test_size) <= 1
