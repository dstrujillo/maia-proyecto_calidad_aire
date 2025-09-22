import math

#import numpy as np
#import pandas as pd
from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient) -> None:
    # Given

    # When
    response = client.get(
        "http://localhost:8001/api/v1/air-quality?lat=40.7128&lon=-74.006",
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["aqi_health_analysis"]
    assert prediction_data["risk_prediction"] 
