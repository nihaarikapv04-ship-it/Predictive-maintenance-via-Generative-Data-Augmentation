from diagnose.gan.physics import bearing_defect_frequencies, DEPLOYMENT_BPFO, DEPLOYMENT_BPFI, DEPLOYMENT_BSF
from diagnose.fusion.model import FusionLSTM
from prescribe.parser import PrescriptionParser


def test_bearing_defect_frequencies_are_computed():
    bpfo, bpfi, bsf = bearing_defect_frequencies(9, 24.0, 0.0075, 0.025, 0.0)
    assert bpfo > 0
    assert bpfi > 0
    assert bsf > 0


def test_deployment_constants_are_separate():
    assert DEPLOYMENT_BPFO == 74.6
    assert DEPLOYMENT_BPFI == 117.4
    assert DEPLOYMENT_BSF == 51.2


def test_fusion_model_returns_mean_and_variance():
    model = FusionLSTM()
    mean_score, variance = model.predict_with_mc_dropout([0.0] * 512, [1.0, 2.0, 3.0, 4.0, 5.0], passes=5)
    assert mean_score >= 0
    assert variance >= 0


def test_prescription_parser_requires_sections():
    parser = PrescriptionParser()
    sections = parser.parse("Immediate Action: Isolate motor\nRepair Protocol: Inspect bearings\nPreventive Schedule: Recheck in 30 days")
    assert sections["Immediate Action"].startswith("Isolate")
    assert sections["Repair Protocol"].startswith("Inspect")
    assert sections["Preventive Schedule"].startswith("Recheck")
