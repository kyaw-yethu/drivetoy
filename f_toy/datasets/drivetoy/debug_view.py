"""Top-down debug view of a simulated clip (cv2). For checking the
simulator by eye; never used as training data."""
import cv2
import numpy as np

from .sim import BALL_G, BALL_R, BALL_Y, PED_FLASH, PED_WALK

_SURF = dict(road=(60, 60, 60), sidewalk=(150, 150, 150), school=(40, 40, 150), island=(90, 120, 90))
_MARK = dict(double_yellow=(0, 200, 230), solid_white=(230, 230, 230), dashed_white=(200, 200, 200), edge=(200, 200, 200),
             stop_line=(255, 255, 255), crosswalk=(220, 220, 220))
_VEH = [(200, 80, 60), (60, 160, 60), (60, 60, 200), (180, 180, 40), (180, 60, 180), (40, 170, 200),
        (120, 120, 220), (90, 90, 90), (200, 200, 200), (30, 120, 220), (150, 100, 60), (100, 200, 140)]


def render_topdown(m, rec, f, size=512, span=110.0, center=None, heading_up=True):
    ego = rec["ego"][f]
    c = np.array(center if center is not None else ego[:2], float)
    ang = (np.pi / 2 - ego[2]) if heading_up else 0.0
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    sc = size / span

    def tf(p):
        q = (np.asarray(p)[..., :2] - c) @ R.T
        q = q - np.array([0.0, span * 0.3]) if heading_up else q
        return np.stack([size / 2 + q[..., 0] * sc, size / 2 - q[..., 1] * sc], -1).astype(np.int32)

    img = np.full((size, size, 3), (70, 110, 70), np.uint8)
    for kind in ("road", "sidewalk", "school"):
        for sf in m.surfaces:
            if sf["kind"] == kind:
                cv2.fillPoly(img, [tf(sf["poly"])], _SURF[kind])
    for b in m.buildings:
        cx, cy = b["center"]
        dx, dy, _ = b["size"] / 2
        h = b["heading"]
        Rb = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
        poly = (np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]]) @ Rb.T) + np.array([cx, cy])
        cv2.fillPoly(img, [tf(poly)], (120, 100, 90))
    for mk in m.markings:
        st = mk["style"]
        if st == "crosswalk":
            cv2.polylines(img, [tf(mk["pts"])], True, _MARK[st], 1)
            continue
        if st.startswith("arrow"):
            p = tf(mk["pts"])
            cv2.arrowedLine(img, tuple(map(int, p[0])), tuple(map(int, p[1])), (230, 230, 230), 1, tipLength=0.4)
            continue
        col = _MARK.get(st, (255, 255, 255))
        pts = tf(mk["pts"])
        if st == "dashed_white":
            d = mk["pts"]
            L = np.linalg.norm(d[-1] - d[0])
            for t in np.arange(0, L, 9.0):
                a, b = d[0] + (d[-1] - d[0]) * t / L, d[0] + (d[-1] - d[0]) * min(t + 3.0, L) / L
                cv2.line(img, tuple(map(int, tf(a))), tuple(map(int, tf(b))), col, 1)
        else:
            cv2.polylines(img, [pts], False, col, 2 if st in ("stop_line", "double_yellow") else 1)
    for st in m.statics:
        cx, cy = st["pos"]
        dx, dy = st["size"][0] / 2, st["size"][1] / 2
        h = st["heading"]
        Rb = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
        poly = (np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]]) @ Rb.T) + np.array([cx, cy])
        cv2.fillPoly(img, [tf(poly)], (0, 140, 255) if st["kind"] == "cone" else (110, 110, 140))
    # signals
    if m.groups:
        for h in m.signal_heads:
            if h["kind"] not in ("veh", "veh_near", "ped"):
                continue
            gi = m.groups.index(h["group"])
            ball, arrow, ped, cd = rec["signals"][f, gi]
            if h["kind"] in ("veh", "veh_near"):
                col = {BALL_R: (0, 0, 255), BALL_Y: (0, 220, 255), BALL_G: (0, 255, 0)}[int(ball)]
                if arrow and h.get("arrow"):
                    col = (255, 255, 0)
                cv2.circle(img, tuple(map(int, tf(h["pos"]))), 4, col, -1)
            else:
                col = {PED_WALK: (255, 255, 255), PED_FLASH: (0, 128, 255)}.get(int(ped), (0, 0, 180))
                cv2.rectangle(img, tuple(map(int, tf(h["pos"]) - 2)), tuple(map(int, tf(h["pos"]) + 2)), col, -1)
    ag, lamps, ex = rec["agents"][f], rec["lamps"][f], rec["extra"][f]
    for a in range(ag.shape[0]):
        if ag[a, 9] <= 0:
            continue
        typ, x, y, h, v = ag[a, :5]
        L, W = ag[a, 6], ag[a, 7]
        if typ >= 5:
            cv2.circle(img, tuple(map(int, tf([x, y]))), max(2, int(0.4 * sc)), (255, 200, 120) if typ == 5 else (255, 120, 200), -1)
            hp = np.array([x, y]) + 0.9 * np.array([np.cos(h), np.sin(h)])
            cv2.line(img, tuple(map(int, tf([x, y]))), tuple(map(int, tf(hp))), (255, 255, 255), 1)
            continue
        c_, s_ = np.cos(h), np.sin(h)
        exv, eyv = np.array([c_, s_]) * L / 2, np.array([-s_, c_]) * W / 2
        p = np.array([x, y])
        poly = np.stack([p + exv + eyv, p - exv + eyv, p - exv - eyv, p + exv - eyv])
        col = (255, 255, 255) if a == 0 else _VEH[a % len(_VEH)]
        cv2.fillPoly(img, [tf(poly)], col)
        if lamps[a, 0]:
            cv2.line(img, tuple(map(int, tf(p - exv + eyv))), tuple(map(int, tf(p - exv - eyv))), (0, 0, 255), 2)
        if lamps[a, 1]:
            cv2.circle(img, tuple(map(int, tf(p + exv + eyv))), 2, (0, 200, 255), -1)
        if lamps[a, 2]:
            cv2.circle(img, tuple(map(int, tf(p + exv - eyv))), 2, (0, 200, 255), -1)
        cv2.putText(img, f"{a}", tuple(map(int, tf(p) + np.array([4, -4]))), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)
    cv2.putText(img, f"{m.template} f={f}", (6, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    return img


def save_video(m, rec, path, fps=5, **kw):
    frames = [render_topdown(m, rec, f, **kw) for f in range(rec["agents"].shape[0])]
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vw.write(fr)
    vw.release()
