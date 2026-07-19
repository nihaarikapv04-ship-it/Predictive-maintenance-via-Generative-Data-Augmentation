from edge.orchestrator import ODPOrchestrator
from observe.camera import CameraCapture
from observe.imu import IMUReader


def test_camera_and_imu_honor_environment_toggle(monkeypatch):
    monkeypatch.setenv("MOTORGUARD_HARDWARE_MODE", "real")
    camera = CameraCapture(simulate=None)
    imu = IMUReader(simulate=None)
    assert camera.simulate is False
    assert imu.simulate is False


def test_orchestrator_uses_environment_toggle(monkeypatch):
    monkeypatch.setenv("MOTORGUARD_HARDWARE_MODE", "simulate")
    orchestrator = ODPOrchestrator(simulate=None)
    assert orchestrator.simulate is True
