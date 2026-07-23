import json
import sys

from scripts import build_leaderboard
from tests.test_benchmark_result import _valid_result


def test_builder_publishes_only_contract_validated_results(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "valid.json").write_text(json.dumps(_valid_result()))
    invalid = _valid_result()
    invalid["refusal_rate_total"] = 2.0
    (results_dir / "invalid.json").write_text(json.dumps(invalid))

    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- BENCH:START -->\nold\n<!-- BENCH:END -->\nafter\n")
    leaderboard = tmp_path / "leaderboard.json"
    monkeypatch.setattr(build_leaderboard, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(build_leaderboard, "README_PATH", readme)
    monkeypatch.setattr(build_leaderboard, "LEADERBOARD_JSON", leaderboard)
    monkeypatch.setattr(build_leaderboard, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_leaderboard.py"])

    assert build_leaderboard.main() == 0

    published = json.loads(leaderboard.read_text())
    assert [row["model"] for row in published] == ["org/abliterated-model"]
    assert published[0]["_source"] == "valid.json"


def test_check_mode_reports_stale_outputs_without_writing(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "valid.json").write_text(json.dumps(_valid_result()))
    readme = tmp_path / "README.md"
    original_readme = "before\n<!-- BENCH:START -->\nold\n<!-- BENCH:END -->\nafter\n"
    readme.write_text(original_readme)
    leaderboard = tmp_path / "leaderboard.json"
    original_leaderboard = "[]\n"
    leaderboard.write_text(original_leaderboard)
    monkeypatch.setattr(build_leaderboard, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(build_leaderboard, "README_PATH", readme)
    monkeypatch.setattr(build_leaderboard, "LEADERBOARD_JSON", leaderboard)
    monkeypatch.setattr(build_leaderboard, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_leaderboard.py", "--check"])

    assert build_leaderboard.main() == 1
    assert readme.read_text() == original_readme
    assert leaderboard.read_text() == original_leaderboard
