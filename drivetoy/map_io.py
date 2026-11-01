"""The ``map`` label (docs/revision-history.md, version 3, Issue 7): a sample's static road map as JSON.

``map_to_dict(m)`` keeps every attribute of the RoadMap and its lanes (centrelines with their arc
length, heading and curvature, successors, neighbours, speed limits and zones, solid-line
intervals, stop lines with their control and signal group, crosswalks, conflict areas, surfaces,
statics, signs, signal heads), tagging NumPy arrays, tuples and tuple-keyed dicts so that
``map_from_dict`` rebuilds a map on which ``rules.check`` and the labels give the simulator's results."""
import numpy as np

from .roads import Lane, RoadMap

VERSION = 1


def _enc(x):
    if isinstance(x, np.ndarray):
        return {"__nd__": x.tolist(), "dtype": str(x.dtype)}
    if isinstance(x, tuple):
        return {"__tuple__": [_enc(v) for v in x]}
    if isinstance(x, dict):
        if all(isinstance(k, str) for k in x):
            return {k: _enc(v) for k, v in x.items()}
        return {"__items__": [[_enc(k), _enc(v)] for k, v in x.items()]}
    if isinstance(x, list):
        return [_enc(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _dec(x):
    if isinstance(x, list):
        return [_dec(v) for v in x]
    if not isinstance(x, dict):
        return x
    if "__nd__" in x:
        return np.array(x["__nd__"], dtype=x["dtype"])
    if "__tuple__" in x:
        return tuple(_dec(v) for v in x["__tuple__"])
    if "__items__" in x:
        return {_dec(k): _dec(v) for k, v in x["__items__"]}
    return {k: _dec(v) for k, v in x.items()}


def map_to_dict(m):
    d = {k: _enc(v) for k, v in m.__dict__.items() if k != "lanes"}
    d["lanes"] = [{k: _enc(v) for k, v in ln.__dict__.items() if not k.startswith("_")} for ln in m.lanes]
    d["map_version"] = VERSION
    return d


def map_from_dict(d):
    m = RoadMap.__new__(RoadMap)
    for k, v in d.items():
        if k == "lanes":
            lanes = []
            for ld in v:
                ln = Lane.__new__(Lane)
                ln.__dict__.update({kk: _dec(vv) for kk, vv in ld.items()})
                lanes.append(ln)
            m.lanes = lanes
        elif k != "map_version":
            setattr(m, k, _dec(v))
    return m


def save_map(path, m):
    """Write the map label as gzipped JSON (``maps/<n>.json.gz``; about a tenth of plain JSON)."""
    import gzip
    import json
    with gzip.open(path, "wt") as f:
        json.dump(map_to_dict(m), f)


def load_map(path):
    import gzip
    import json
    with gzip.open(path, "rt") as f:
        return map_from_dict(json.load(f))
