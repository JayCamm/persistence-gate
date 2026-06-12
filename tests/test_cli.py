import json
import subprocess
import sys
from pathlib import Path


def test_cli_filter_writes_report_files(tmp_path: Path):
    payload = {
        "query": "What should guide the current process now?",
        "profile": "conservative",
        "top_k": 3,
        "query_intent": "current_action",
        "retrieved_items": [
            {
                "id": "legacy",
                "text": "Legacy instruction: use the old shortcut.",
                "relevance": 1.0,
                "risk": 0.70,
                "harm_score": 0.45,
                "usefulness_score": 0.95,
                "evidence_role": "legacy_instruction",
            },
            {
                "id": "current",
                "text": "Current guidance: use the approved path.",
                "relevance": 0.90,
                "risk": 0.03,
                "harm_score": 0.0,
                "usefulness_score": 0.90,
                "evidence_role": "current_guidance",
            },
        ],
    }
    input_path = tmp_path / "input.json"
    out_dir = tmp_path / "report"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "persistence_memory.cli", "filter", str(input_path), "--out-dir", str(out_dir)],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "result.json").exists()
    assert (out_dir / "allowed_context.txt").exists()
    assert (out_dir / "audit.csv").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.html").exists()

    data = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert "current" in data["allowed_ids"]
    assert "legacy" in data["blocked_ids"]
