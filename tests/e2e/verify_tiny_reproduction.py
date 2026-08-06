"""Run the pinned tiny model twice, compare weights, then replay its manifest."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("tiny_repro.toml")


def _run(*arguments: str, cwd: Path) -> None:
    subprocess.run(
        ["uv", "run", "--project", str(ROOT), "abliterix", *arguments],
        cwd=cwd,
        check=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="abliterix-e2e-") as tmp:
        root = Path(tmp)
        manifests: list[dict] = []
        for index in (1, 2):
            run_dir = root / f"run{index}"
            run_dir.mkdir()
            output = run_dir / "model"
            _run(
                "--config",
                str(CONFIG),
                "--non-interactive-output-dir",
                str(output),
                "--optimization.checkpoint-dir",
                str(run_dir / "checkpoints"),
                "--overwrite-checkpoint",
                cwd=run_dir,
            )
            manifests.append(
                json.loads((output / "reproduce" / "reproduce.json").read_text())
            )

        if manifests[0]["weights"] != manifests[1]["weights"]:
            raise RuntimeError(
                "Two independent tiny-model runs produced different weight hashes: "
                f"{manifests[0]['weights']} != {manifests[1]['weights']}"
            )
        source_is_clean = not subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if source_is_clean and not all(
            manifest.get("reproducible") for manifest in manifests
        ):
            raise RuntimeError("A pinned deterministic E2E run was not reproducible.")

        _run(
            "--reproduce",
            str(root / "run1/model/reproduce/reproduce.json"),
            cwd=root / "run1",
        )
        print("Tiny-model weight hashes and exact manifest replay verified.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
