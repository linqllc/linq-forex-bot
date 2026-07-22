from pathlib import Path


def test_required_project_files_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    required = [
        root / "README.md",
        root / "CONTRIBUTING.md",
        root / "SECURITY.md",
        root / "pyproject.toml",
        root / ".github" / "workflows" / "ci.yml",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    assert not missing, f"Missing required project files: {missing}"
