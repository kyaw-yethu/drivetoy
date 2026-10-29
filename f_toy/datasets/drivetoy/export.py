"""Hand a simulated clip to the Blender renderer: one JSON scene file
(static map geometry, look, camera, and the per-frame states of the
rendered frames). Plain lists only, so the Blender-side script needs nothing
from this package."""
import json

import numpy as np

from .sim import BALL_G


def _l(a):
    return np.asarray(a, dtype=float).round(4).tolist()


def scene_dict(result, branch, cfg, seed):
    m = result["map"]
    rec = result["branches"][branch]
    rng = np.random.RandomState(seed + 7919)
    T = cfg["FRAMES"]
    tod = rng.choice(["day", "day", "day", "dusk"])
    sun_el = float(rng.uniform(25, 65) if tod == "day" else rng.uniform(3, 10))
    sun_az = float(rng.uniform(0, 360))
    if tod == "dusk" and cfg.get("DT_V2"):
        # v2: a low sun ahead of the camera backlights the asphalt, which Cycles renders noisy at
        # 16 samples: keep it DT_DUSK_SUN_AWAY degrees from every heading of the ego in the clip
        heads = np.degrees(np.asarray(rec["ego"][:T, 2]))
        for _ in range(50):
            if np.min(np.abs((sun_az - heads + 180.0) % 360.0 - 180.0)) >= cfg["DT_DUSK_SUN_AWAY"]:
                break
            sun_az = float(rng.uniform(0, 360))
    near = lambda h: False
    ego_leg = None
    for f in range(T):
        lid = int(rec["extra"][f, 0, 0])
        ego_leg = getattr(m.lanes[lid], "leg", None) if lid >= 0 else None
        if ego_leg is not None:
            break
    # v2: only the ego's own approach has its heads moved nearer (pos_near, see roads._four_way)
    hpos = lambda h: h["pos_near"] if ("pos_near" in h and h["leg"] == ego_leg) else h["pos"]
    if cfg.get("DT_HIDE_NEAR_HEADS") and m.signal_heads:
        # v2: vehicle heads on the ego's near side of the junction (those of the opposite and the
        # left-hand approach, on the two near corners) are not rendered; the far ones stay
        for f in range(T):
            lid = int(rec["extra"][f, 0, 0])
            leg = getattr(m.lanes[lid], "leg", None) if lid >= 0 else None
            if leg is not None:
                hd = np.pi / 2 + leg * np.pi / 2
                d = np.array([np.cos(hd), np.sin(hd)])
                c = np.asarray(m.intersection["center"], float)
                near = lambda h, d=d, c=c: h["kind"] != "ped" and float(np.dot(np.asarray(hpos(h)[:2]) - c, d)) < 0.0
                break
    look = dict(
        time=str(tod),
        sun_elevation=sun_el,
        sun_azimuth=sun_az,
        fog=float(rng.uniform(0.0, 0.5) if rng.rand() < 0.3 else 0.0),
        grass=int(rng.randint(4)), asphalt=float(rng.uniform(0.06, 0.14)),
        building_seed=int(rng.randint(1 << 30)),
    )
    cam = dict(height=float(rng.uniform(*cfg["CAM_HEIGHT"])), pitch=float(rng.uniform(*cfg["CAM_PITCH"])),
               fov=float(cfg["CAM_FOV"]), forward=float(cfg["CAM_FORWARD"]))
    return dict(
        template=m.template, img=cfg["IMG"], frames=T, cue_scale=float(cfg["DT_CUE_SCALE"]),
        signal_scale=float(cfg.get("DT_SIGNAL_SCALE", cfg["DT_CUE_SCALE"])),
        signal_layout=str(cfg.get("DT_SIGNAL_LAYOUT", "vertical")),
        pole_head=bool(cfg.get("DT_POLE_HEAD", True)),
        indicator_scale=float(cfg.get("DT_INDICATOR_SCALE", cfg["DT_CUE_SCALE"])),
        lamp_emit=dict(cfg["DT_LAMP_EMIT"]), view=list(cfg["DT_VIEW"]), seed=int(seed), branch=int(branch),
        signal_colors={k: list(v) for k, v in cfg.get("DT_SIGNAL_COLORS", {}).items()},
        groups=list(m.groups), protected_left=bool(m.protected_left), countdown=bool(m.countdown),
        surfaces=[dict(kind=s["kind"], poly=_l(s["poly"])) for s in m.surfaces],
        markings=[dict(style=k["style"], pts=_l(k["pts"])) for k in m.markings],
        buildings=[dict(center=_l(b["center"]), size=_l(b["size"]), heading=float(b["heading"])) for b in m.buildings],
        statics=[dict(kind=s["kind"], pos=_l(s["pos"]), heading=float(s["heading"]), size=_l(s["size"]),
                      color=int(i * 7 % 12)) for i, s in enumerate(m.statics)],
        signs=[dict(kind=s["kind"], pos=_l(s["pos"]), heading=float(s["heading"]), value=s.get("value"),
                    size=float(s.get("size", 1.0))) for s in m.signs],
        heads=[dict(kind=h["kind"], group=h["group"], pos=_l(hpos(h)), heading=float(h["heading"]),
                    arrow=bool(h.get("arrow", False)), countdown=bool(h.get("countdown", False)), leg=int(h["leg"]))
               for h in m.signal_heads if not near(h)],
        look=look, camera=cam,
        agents=_l(rec["agents"][:T]), lamps=np.asarray(rec["lamps"][:T]).astype(int).tolist(),
        signals=_l(rec["signals"][:T]), ego=_l(rec["ego"][:T]),
        colors={str(k): int(v) for k, v in result["colors"].items()},
    )


def write_scene(path, result, branch, cfg, seed):
    with open(path, "w") as f:
        json.dump(scene_dict(result, branch, cfg, seed), f)
