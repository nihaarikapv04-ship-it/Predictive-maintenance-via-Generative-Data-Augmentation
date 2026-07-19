import json

from edge.api import app


def test_dashboard_page_serves_html():
    client = app.test_client()
    response = client.get("/dashboard")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "MotorGuard Dashboard" in body


def test_overview_endpoint_returns_pipeline_snapshot():
    client = app.test_client()
    response = client.get("/overview")

    assert response.status_code == 200
    payload = json.loads(response.get_data(as_text=True))
    assert "health" in payload
    assert "config" in payload
    assert "odp" in payload
