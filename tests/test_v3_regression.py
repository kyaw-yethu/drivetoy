"""v3 must leave v2 runs unchanged: drive_v2s val samples re-simulate exactly from their stored
cfg (no DT_V3). python3 tests/test_v3_regression.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import v2s_checks

V2S_VAL = Path("/root/projects/drive_v2s/val")


def test_v2s_val_resimulates():
    v2s_checks.ROOT = V2S_VAL
    v2s_checks.META = json.loads((V2S_VAL / "meta.json").read_text())
    assert "DT_V3" not in v2s_checks.META["cfg"]
    import numpy as np
    tmpl = np.load(V2S_VAL / "annotations.npz")["template"]
    for t in range(4):
        n = int(np.where(tmpl == t)[0][0])
        assert v2s_checks.resim(n)[1], n


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
