import pytest
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'message' in data


def test_predict_missing_data(client):
    """Test predict endpoint with missing data."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_invalid_hours(client):
    """Test predict endpoint with invalid hours."""
    response = client.post('/predict', json={
        'hours_studied': 'invalid',
        'exam_difficulty': 'Medium'
    })
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_invalid_difficulty(client):
    """Test predict endpoint with invalid difficulty."""
    response = client.post('/predict', json={
        'hours_studied': 5.0,
        'exam_difficulty': 'Invalid'
    })
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_with_model(client):
    """Test predict endpoint with valid model (if available)."""
    response = client.post('/predict', json={
        'hours_studied': 5.0,
        'exam_difficulty': 'Medium'
    })
    
    # Should either succeed (200) or fail due to missing model (500)
    assert response.status_code in [200, 500]
    
    data = json.loads(response.data)
    if response.status_code == 200:
        assert 'predicted_score' in data
        assert 'input' in data
    else:
        assert 'error' in data


def test_model_info_with_model(client):
    """Test model info endpoint with valid model (if available)."""
    response = client.get('/model_info')
    
    # Should either succeed (200) or fail due to missing model (500)
    assert response.status_code in [200, 500]
    
    data = json.loads(response.data)
    if response.status_code == 200:
        assert 'model_type' in data
    else:
        assert 'error' in data
