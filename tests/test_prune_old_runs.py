import os
import shutil
import time
from pathlib import Path

from src.gradio_ui import _prune_old_runs


def create_run_dirs(root: Path, count: int, base_time: float = None):
    if base_time is None:
        base_time = time.time()
    dirs = []
    for i in range(count):
        d = root / f"run_{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        # set mtime spaced by i seconds so higher i -> newer
        mtime = base_time + i
        os.utime(d, (mtime, mtime))
        dirs.append(d)
    return dirs


def test_prune_keeps_newest(tmp_path):
    root = tmp_path / "gradio_runs"
    root.mkdir()
    dirs = create_run_dirs(root, 5)
    # Keep only the 2 newest directories
    _prune_old_runs(root, keep=2)
    remaining = sorted([p.name for p in root.iterdir() if p.is_dir()])
    assert len(remaining) == 2
    assert set(remaining) == {dirs[-1].name, dirs[-2].name}


def test_prune_env_var_override_and_disable(tmp_path, monkeypatch):
    root = tmp_path / "gradio_runs"
    root.mkdir()
    dirs = create_run_dirs(root, 6)
    # set env var to 3 -> keep 3 newest
    monkeypatch.setenv("FORT_GRADIO_RETENTION_KEEP", "3")
    _prune_old_runs(root)
    remaining = sorted([p.name for p in root.iterdir() if p.is_dir()])
    assert len(remaining) == 3
    assert set(remaining) == {d.name for d in dirs[-3:]}

    # disable pruning via 0 -> nothing deleted
    # remove existing dirs and recreate
    for p in list(root.iterdir()):
        if p.is_dir():
            shutil.rmtree(p)
    dirs = create_run_dirs(root, 4)
    monkeypatch.setenv("FORT_GRADIO_RETENTION_KEEP", "0")
    _prune_old_runs(root)
    remaining = sorted([p.name for p in root.iterdir() if p.is_dir()])
    assert len(remaining) == 4
