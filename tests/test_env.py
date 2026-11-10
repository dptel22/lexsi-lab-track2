"""Smoke test suite verifying Python runtime, vendor directory accessibility, and Stage 1 dependencies."""

import importlib
import sys
from pathlib import Path


def test_python_version_is_3_12():
    """Verify runtime Python interpreter is version 3.12.x."""
    assert sys.version_info.major == 3, f"Expected Python 3, got {sys.version_info.major}"
    assert sys.version_info.minor == 12, f"Expected Python 3.12, got 3.{sys.version_info.minor}"


def test_vendor_repositories_accessible():
    """Verify vendor directory and all 4 official Lexsi-Labs repositories exist and contain valid files."""
    repo_root = Path(__file__).resolve().parent.parent
    vendor_dir = repo_root / "vendor"
    assert vendor_dir.exists(), f"vendor directory missing at {vendor_dir}"
    assert vendor_dir.is_dir(), f"{vendor_dir} is not a directory"

    required_repos = ["CuratorKIT", "CircuitKIT", "SafeTune", "AuditKIT"]
    for repo_name in required_repos:
        repo_path = vendor_dir / repo_name
        assert repo_path.exists(), f"Vendor repository {repo_name} missing at {repo_path}"
        assert (repo_path / "pyproject.toml").is_file(), f"{repo_name} missing pyproject.toml"
        assert (repo_path / ".git").exists(), f"{repo_name} missing .git tracking"


def test_curatorkit_import_and_contract():
    """Verify CuratorKIT is installed, importable, and provides core classes."""
    curatorkit = importlib.import_module("curatorkit")
    assert hasattr(curatorkit, "__version__"), "curatorkit has no __version__"
    assert curatorkit.__version__ == "0.1.21", f"Unexpected curatorkit version: {curatorkit.__version__}"

    # Verify core interface exports
    from curatorkit import Curator, CuratorConfig, DataSample, JSONLReader

    assert Curator is not None
    assert CuratorConfig is not None
    assert DataSample is not None
    assert JSONLReader is not None


def test_auditkit_import_and_contract():
    """Verify AuditKIT is installed, importable, and provides core classes."""
    auditkit = importlib.import_module("auditkit")
    assert hasattr(auditkit, "__version__"), "auditkit has no __version__"
    assert auditkit.__version__ == "1.1.6", f"Unexpected auditkit version: {auditkit.__version__}"

    # Verify core interface exports
    from auditkit import Evaluator, Scenario, Sample, MetricDelta

    assert Evaluator is not None
    assert Scenario is not None
    assert Sample is not None
    assert MetricDelta is not None


def test_src_scaffold_and_pythonpath():
    """Verify src/ structure exists and is configured in sys.path via pyproject.toml."""
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    assert src_dir.exists() and src_dir.is_dir(), f"src directory missing at {src_dir}"
    assert (src_dir / "__init__.py").is_file(), f"src/__init__.py missing at {src_dir / '__init__.py'}"

    # Verify src directory is resolved in sys.path
    normalized_sys_path = [str(Path(p).resolve()).lower() for p in sys.path if p]
    assert str(src_dir.resolve()).lower() in normalized_sys_path, (
        f"{src_dir.resolve()} not found in sys.path: {sys.path}"
    )
