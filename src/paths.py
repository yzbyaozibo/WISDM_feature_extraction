"""Resolve repository paths independently of the caller's working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def project_path(relative_path):
    return PROJECT_ROOT / relative_path

def output_path(relative_dir, filename):
    path = project_path(relative_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
