import os
from pathlib import Path

import rag


def test_resolve_repo_path_uses_script_location(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    resolved = rag.resolve_repo_path("docs/motor_manual.pdf")
    assert resolved.exists()
    assert resolved.name == "motor_manual.pdf"
    assert resolved.parent.name == "docs"
