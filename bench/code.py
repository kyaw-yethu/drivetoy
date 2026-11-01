"""Selects the drivetoy package that built the split being scored.

drive_v3 was built by commit f02bf91. Its package was copied out with
    git archive f02bf91 f_toy/datasets/drivetoy | tar -x -C /root/opt/drivetoy_v3_build
and moved to /root/opt/drivetoy_v3_build/drivetoy, with an __init__.py that leaves out DiskDataset.
"""
import importlib
import sys
from pathlib import Path

DEFAULT = '/root/opt/drivetoy_v3_build'   # drivetoy_v3 commit f02bf91, which built drive_v3


def use(root=DEFAULT):
    """Import drivetoy from ``root`` and return it; fails if another drivetoy is already loaded."""
    root = str(Path(root).resolve())
    loaded = sys.modules.get('drivetoy')
    if loaded is not None:
        if not str(Path(loaded.__file__).resolve()).startswith(root):
            raise RuntimeError(f'drivetoy already imported from {loaded.__file__}, not {root}')
        return loaded
    sys.path.insert(0, root)
    mod = importlib.import_module('drivetoy')
    if not str(Path(mod.__file__).resolve()).startswith(root):
        raise RuntimeError(f'drivetoy resolved to {mod.__file__}, not {root}')
    return mod
