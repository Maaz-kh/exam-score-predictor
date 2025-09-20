from _future_ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(_name_)

# Global model artifact (loaded once at startup)
MODEL_ARTIFACT = None


def load_model(model_path: str = "models/model.joblib") -> Dict[str, Any]:
    """Load the trained model artifact."""
    global MODEL_ARTIFACT
    if MODEL_ARTIFACT is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
        MODEL_ARTIFACT = joblib.load(model_path)
    return MODEL_ARTIFACT


@app.route("/health", methods=["GET"])
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}


@app.route("/predict", methods=["POST"])
def predict() -> Dict[str, Any]:
    """Predict exam score based on hours studied and exam difficulty."""
    try:
        # Load model if not already loaded
        artifact = load_model()
        model = artifact["model"] if isinstance(artifact, dict) else artifact
        feature_names = (
            artifact.get("feature_names", None) if isinstance(artifact, dict) else None
        )

        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        hours = data.get("hours_studied")
        difficulty = data.get("exam_difficulty", "Medium")

        # Validate input
        if hours is None:
            return jsonify({"error": "hours_studied is required"}), 400
        
        try:
            hours = float(hours)
        except (ValueError, TypeError):
            return jsonify({"error": "hours_studied must be a number"}), 400

        if difficulty not in ["Easy", "Medium", "Hard"]:
            return jsonify({"error": "exam_difficulty must be Easy, Medium, or Hard"}), 400

        # Construct feature vector
        base = {
            "hours_studied": hours,
            "difficulty_Easy": 1.0 if difficulty == "Easy" else 0.0,
            "difficulty_Medium": 1.0 if difficulty == "Medium" else 0.0,
            "difficulty_Hard": 1.0 if difficulty == "Hard" else 0.0,
        }
        X = pd.DataFrame([base])

        # Ensure feature order matches training
        if feature_names is not None:
            for col in feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[feature_names]

        # Make prediction
        prediction = float(model.predict(X)[0])
        
        # Clamp prediction to reasonable range (0-100)
        prediction = max(0.0, min(100.0, prediction))

        return jsonify({
            "predicted_score": round(prediction, 2),
            "input": {
                "hours_studied": hours,
                "exam_difficulty": difficulty
            }
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/model_info", methods=["GET"])
def model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    try:
        artifact = load_model()
        model = artifact["model"] if isinstance(artifact, dict) else artifact
        feature_names = (
            artifact.get("feature_names", None) if isinstance(artifact, dict) else None
        )

        return jsonify({
            "model_type": type(model)._name_,
            "feature_names": feature_names,
            "model_params": getattr(model, 'get_params', lambda: {})()
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get model info: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)