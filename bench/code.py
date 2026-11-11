"""Selects the drivetoy code that built the split being scored.

Every drive_v4 split was built by commit c6d3763 (``drivetoy.decision.set_stage`` reproduces the stage of each
plan entry). bench/setup_env.sh copies its drivetoy/, scripts/build.py and plans/ out with
    git archive c6d3763 drivetoy scripts/build.py plans | tar -x -C /root/opt/drivetoy_v4_build
"""
import importlib
import sys
from pathlib import Path

DEFAULT = '/root/opt/drivetoy_v4_build'   # commit c6d3763, which built drive_v4


def _check(mod, root):
    if not str(Path(mod.__file__).resolve()).startswith(root):
        raise RuntimeError(f'{mod.__name__} resolved to {mod.__file__}, not {root}')
    return mod


def use(root=DEFAULT):
    """Import drivetoy from ``root`` and return it; fails if another drivetoy is already loaded."""
    root = str(Path(root).resolve())
    loaded = sys.modules.get('drivetoy')
    if loaded is not None:
        return _check(loaded, root)
    sys.path.insert(0, root)
    return _check(importlib.import_module('drivetoy'), root)


def build(root=DEFAULT):
    """scripts/build.py of ``root``: the plans of the splits and how each plan entry was simulated."""
    use(root)
    root = str(Path(root).resolve())
    loaded = sys.modules.get('build')
    if loaded is not None:
        return _check(loaded, root)
    sys.path.insert(0, str(Path(root) / 'scripts'))
    return _check(importlib.import_module('build'), root)
