"""Configuration, dispatch, and CLI contract tests."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cli_configuration_has_no_runtime_state_fields():
    for name in ("crl", "sac", "td3", "ppo"):
        tree = ast.parse((ROOT / f"jaxgcrl/agents/{name}/{name}.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name.upper())
        fields = [
            n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        ]
        assert not [f for f in fields if f.startswith("_")], name


def test_training_dispatch_uses_explicit_strategy_contract():
    tree = ast.parse((ROOT / "jaxgcrl/train.py").read_text())
    attrs = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert "strategy" in attrs
    assert "train_fn" not in attrs
    assert "_training_state" not in attrs


@pytest.mark.parametrize("subcmd", [[], ["crl"], ["ppo"], ["sac"], ["td3"]])
def test_cli_help_succeeds(subcmd):
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = f"{ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(ROOT)
    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "MPLBACKEND": "Agg",
    }
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *subcmd, "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout.lower()
