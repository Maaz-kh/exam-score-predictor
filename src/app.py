from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
