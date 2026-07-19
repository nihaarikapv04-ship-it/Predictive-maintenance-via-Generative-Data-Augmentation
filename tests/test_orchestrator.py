from edge.orchestrator import ODPOrchestrator


def test_orchestrator_runs_once():
    orchestrator = ODPOrchestrator(simulate=True)
    result = orchestrator.run_once()

    assert "camera" in result
    assert len(result["vibration_features"]) == 5
    assert result["fusion_score"] >= 0
    assert result["fusion_variance"] >= 0
    assert "Immediate Action" in result["prescription"]
