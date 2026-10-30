"""Blender (bpy) renderer for DriveToy clips — run inside Blender, not the venv:

    blender -b --python render_clip.py -- --jobs jobs.txt --gpu 0 [--samples 16]

``jobs.txt`` holds one ``scene.json<TAB>out.npz`` pair per line (scenes come
from ``export.write_scene``). For every job the scene is rebuilt from
procedural low-poly geometry and the clip is rendered from the ego camera
with Cycles on one CUDA device. ``out.npz`` gets ``video`` (T,H,W,3 uint8)
and ``index`` (T,H,W int16): the object index under each pixel —
1+slot for agents, 1000+4*slot+k for a vehicle's lamp k (0 brake, 1 left,
2 right), 2000+i for signal head i, 3000+i for sign i, 4000+i for crosswalk
i's markings, 5000+4*i+k for lamp k (0 red, 1 yellow, 2 green, 3 arrow) of
vehicle signal head i, 0 for everything else.

Geometry is accumulated into a few merged meshes (one per pass index), since
Cycles' per-frame sync cost grows with the object count, not the triangles.

Everything here is plain bpy + numpy; nothing is imported from f_toy.
"""
import json
import math
import os
import sys
import tempfile
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import crown_blocks   # noqa: E402  (no tree crowns in the sightlines)

CAR_COLORS = [(0.55, 0.05, 0.04), (0.02, 0.12, 0.40), (0.85, 0.85, 0.85), (0.05, 0.05, 0.05), (0.45, 0.45, 0.47),
              (0.10, 0.28, 0.12), (0.75, 0.55, 0.08), (0.20, 0.35, 0.55), (0.60, 0.60, 0.62), (0.30, 0.08, 0.20),
              (0.80, 0.30, 0.05), (0.15, 0.15, 0.20)]
SHIRT_COLORS = [(0.7, 0.1, 0.1), (0.1, 0.3, 0.7), (0.9, 0.8, 0.2), (0.1, 0.5, 0.2), (0.85, 0.85, 0.85), (0.15, 0.15, 0.15),
                (0.9, 0.45, 0.1), (0.5, 0.2, 0.6), (0.3, 0.6, 0.8), (0.6, 0.4, 0.3), (0.95, 0.5, 0.6), (0.4, 0.4, 0.1)]
WALL_COLORS = [(0.62, 0.55, 0.45), (0.45, 0.22, 0.16), (0.55, 0.55, 0.55), (0.75, 0.72, 0.66), (0.35, 0.38, 0.42),
               (0.68, 0.60, 0.50)]
GRASS = [(0.12, 0.25, 0.07), (0.18, 0.28, 0.10), (0.25, 0.27, 0.15), (0.10, 0.20, 0.08)]
SUN_COLOR = dict(day=(1.0, 0.97, 0.92), dusk=(1.0, 0.62, 0.35))
VEH_TYPES = (1, 2, 3, 4)
CUE = [1.0]   # lamp / pedestrian-head size multiplier (DT_CUE_SCALE), set per job
SIG = [1.0]   # vehicle signal-head size multiplier (DT_SIGNAL_SCALE)
LAYOUT = ["vertical"]   # lamp layout of the mast-arm heads (DT_SIGNAL_LAYOUT)
POLE_HEAD = [True]      # draw the head on the far-right corner pole (DT_POLE_HEAD)


def _horizontal(h):
    return h["kind"] == "veh" and LAYOUT[0] == "horizontal"


def _half_height(h):
    """Half the housing height of a vehicle head, for the hanger and the mast arm."""
    return (0.36 + 0.1) * SIG[0] / 2 if _horizontal(h) else HOUSING * SIG[0] / 2
IND = [1.0]   # indicator size multiplier (DT_INDICATOR_SCALE)
HOUSING = 0.36 * 3 + 0.1   # height of a three-lamp head at real size
LAMP_ON = dict(brake=6.0, left=10.0, right=10.0, signal=12.0, ped=8.0)   # emission strengths; a scene's
                                                                            # "lamp_emit" overrides them


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------
_mats = {}


def mat(name, color, rough=0.6, metal=0.0, emit=None, strength=0.0):
    key = (name, tuple(np.round(color, 3)))
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1.0)
        b.inputs["Emission Strength"].default_value = strength
    _mats[key] = m
    return m


def lamp_mat(name, color):
    """A lamp material of its own, whose emission is switched per frame."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*[c * 0.2 for c in color], 1.0)
    b.inputs["Emission Color"].default_value = (*color, 1.0)
    b.inputs["Emission Strength"].default_value = 0.0
    b.inputs["Roughness"].default_value = 0.3
    m["_on"] = None
    return m


def set_emit(m, on, strength):
    on = bool(on)
    if m.get("_on") == on:
        return                     # untouched materials need no shader re-sync
    m["_on"] = on
    m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = strength if on else 0.0


def building_mat(color):
    key = ("building", tuple(color))
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new("building")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.8
    tex = nt.nodes.new("ShaderNodeTexBrick")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    tex.inputs["Color1"].default_value = (0.06, 0.08, 0.10, 1)
    tex.inputs["Color2"].default_value = (0.10, 0.12, 0.15, 1)
    tex.inputs["Mortar"].default_value = (*color, 1)
    tex.inputs["Scale"].default_value = 1.0
    tex.inputs["Mortar Size"].default_value = 0.35
    tex.inputs["Brick Width"].default_value = 2.2
    tex.inputs["Row Height"].default_value = 3.0
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    add = nt.nodes.new("ShaderNodeMath")
    nt.links.new(coord.outputs["Object"], sep.inputs[0])
    nt.links.new(sep.outputs["X"], add.inputs[0])
    nt.links.new(sep.outputs["Y"], add.inputs[1])
    nt.links.new(add.outputs[0], comb.inputs["X"])
    nt.links.new(sep.outputs["Z"], comb.inputs["Y"])
    nt.links.new(comb.outputs[0], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    _mats[key] = m
    return m


# ---------------------------------------------------------------------------
# geometry: primitives return (verts, faces); Geo merges them into one object
# ---------------------------------------------------------------------------

def g_box(cx, cy, cz, sx, sy, sz, taper=0.0):
    x0, x1, y0, y1, z0, z1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2
    t, ty = taper * sx / 2, taper * 0.15 * sy
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0 + t, y0 + ty, z1), (x1 - 0.6 * t, y0 + ty, z1), (x1 - 0.6 * t, y1 - ty, z1), (x0 + t, y1 - ty, z1)]
    return v, [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def g_quad(center, heading, w, h, offset=0.0):
    """Vertical quad facing ``heading``."""
    cx, cy, cz = center
    nx, ny = math.cos(heading), math.sin(heading)
    tx, ty = -ny, nx
    cx, cy = cx + nx * offset, cy + ny * offset
    return [(cx - tx * w / 2, cy - ty * w / 2, cz - h / 2), (cx + tx * w / 2, cy + ty * w / 2, cz - h / 2),
            (cx + tx * w / 2, cy + ty * w / 2, cz + h / 2), (cx - tx * w / 2, cy - ty * w / 2, cz + h / 2)], [(0, 1, 2, 3)]


def g_disc(center, heading, r, n=12):
    cx, cy, cz = center
    tx, ty = -math.sin(heading), math.cos(heading)
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    v = [(cx, cy, cz)] + [(cx + tx * r * math.cos(t), cy + ty * r * math.cos(t), cz + r * math.sin(t)) for t in a]
    return v, [(0, i + 1, (i + 1) % n + 1) for i in range(n)]


def g_cyl(x, y, z0, z1, r, n=8):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    v = [(x + r * math.cos(t), y + r * math.sin(t), z) for z in (z0, z1) for t in a]
    return v, [tuple(range(n))[::-1], tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]


def _ccw(poly):
    p = np.asarray(poly, float)
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    return p if area > 0 else p[::-1]


def g_poly(poly, z):
    p = _ccw(poly)
    return [(q[0], q[1], z) for q in p], [tuple(range(len(p)))]


def g_slab(poly, z0, z1):
    p = _ccw(poly)
    n = len(p)
    v = [(q[0], q[1], z0) for q in p] + [(q[0], q[1], z1) for q in p]
    return v, [tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]


def g_strip(pts, width, z):
    pts = np.asarray(pts, float)
    v, f = [], []
    for i, p in enumerate(pts):
        d = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        d = d / (np.linalg.norm(d) + 1e-9)
        nrm = np.array([-d[1], d[0]]) * width / 2
        v += [(*(p - nrm), z), (*(p + nrm), z)]
        if i:
            f.append((2 * i - 2, 2 * i, 2 * i + 1, 2 * i - 1))
    return v, f


def g_blob(x, y, z, r):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    v = v / np.linalg.norm(v[0]) * r * np.array([1.0, 1.0, 0.85]) + np.array([x, y, z])
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    return [tuple(q) for q in v], f


def xf(vf, x=0.0, y=0.0, z=0.0, rz=0.0):
    v, f = vf
    c, s = math.cos(rz), math.sin(rz)
    return [(x + c * a - s * b, y + s * a + c * b, z + h) for a, b, h in v], f


class Geo:
    def __init__(self):
        self.v, self.f, self.mi, self.mats = [], [], [], []

    def add(self, vf, material):
        v, f = vf
        if not f:
            return
        if material not in self.mats:
            self.mats.append(material)
        k = self.mats.index(material)
        base = len(self.v)
        self.v.extend(v)
        self.f.extend(tuple(base + i for i in face) for face in f)
        self.mi.extend([k] * len(f))

    def obj(self, name, index=0, parent=None):
        if not self.f:
            return None
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(map(float, p)) for p in self.v], [], self.f)
        for m in self.mats:
            me.materials.append(m)
        me.polygons.foreach_set("material_index", self.mi)
        me.update()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        ob.pass_index = index
        if parent is not None:
            ob.parent = parent
        return ob


def text_obj(name, body, size, material, loc, rot_z, index=0, parent=None):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    ob = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(material)
    ob.pass_index = index
    ob.location = loc
    ob.rotation_euler = (math.pi / 2, 0, rot_z + math.pi / 2)
    if parent is not None:
        ob.parent = parent
    return ob


# ---------------------------------------------------------------------------
# static scene
# ---------------------------------------------------------------------------

def build_static(sc, rng):
    look = sc["look"]
    geos = {}

    def G(index):
        return geos.setdefault(index, Geo())

    asphalt = mat("asphalt", (look["asphalt"],) * 3, rough=0.9)
    sidewalk = mat("sidewalk", (0.42, 0.41, 0.39), rough=0.85)
    school = mat("school", (0.45, 0.07, 0.05), rough=0.8)
    white = mat("paint_w", (0.85, 0.85, 0.85), rough=0.5)
    yellow = mat("paint_y", (0.85, 0.62, 0.05), rough=0.5)
    G(0).add(g_poly([(-600, -600), (600, -600), (600, 600), (-600, 600)], -0.02), mat("grass", GRASS[look["grass"]], rough=1.0))
    for s in sc["surfaces"]:
        if s["kind"] == "road":
            G(0).add(g_poly(s["poly"], 0.0), asphalt)
        elif s["kind"] == "sidewalk":
            G(0).add(g_slab(s["poly"], 0.0, 0.15), sidewalk)
        elif s["kind"] == "school":
            G(0).add(g_poly(s["poly"], 0.006), school)
    cw_i = 0
    z = 0.012
    for mk in sc["markings"]:
        st, pts = mk["style"], np.asarray(mk["pts"], float)
        if st in ("solid_white", "edge"):
            G(0).add(g_strip(pts, 0.15, z), white)
        elif st == "stop_line":
            G(0).add(g_strip(pts, 0.45, z), white)
        elif st == "school_triangle":   # filled blue, on the red zone pavement
            G(0).add(g_poly(pts, z), mat("school_tri", (0.04, 0.16, 0.7), rough=0.5))
        elif st == "double_yellow":
            d = pts[-1] - pts[0]
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            for sgn in (-1, 1):
                G(0).add(g_strip(pts + sgn * 0.13 * n, 0.12, z), yellow)
        elif st == "dashed_white":
            a, b = pts[0], pts[-1]
            L = np.linalg.norm(b - a)
            u = (b - a) / (L + 1e-9)
            for t in np.arange(0.0, L, 9.0):
                G(0).add(g_strip(np.stack([a + u * t, a + u * min(t + 3.0, L)]), 0.13, z), white)
        elif st == "crosswalk":
            _zebra(G(4000 + cw_i), pts, white, z)
            cw_i += 1
        elif st.startswith("arrow"):
            _arrow(G(0), st, pts, white, z)
    for b in sc["buildings"]:
        cx, cy = b["center"]
        sx, sy, sz = b["size"]
        G(0).add(xf(g_box(0, 0, sz / 2, sx, sy, sz), cx, cy, 0, b["heading"]), building_mat(WALL_COLORS[rng.randint(len(WALL_COLORS))]))
        G(0).add(xf(g_box(0, 0, sz + 0.15, sx + 0.3, sy + 0.3, 0.3), cx, cy, 0, b["heading"]), mat("roof", (0.2, 0.2, 0.22)))
    _trees(G(0), sc, rng)
    for i, st in enumerate(sc["statics"]):
        x, y = st["pos"]
        if st["kind"] == "cone":
            o = mat("cone", (0.9, 0.3, 0.02))
            n = 8
            a = np.linspace(0, 2 * math.pi, n, endpoint=False)
            G(0).add(([(x + 0.2 * math.cos(t), y + 0.2 * math.sin(t), 0) for t in a] + [(x, y, 0.75)],
                      [(k, (k + 1) % n, n) for k in range(n)]), o)
            G(0).add(g_box(x, y, 0.02, 0.45, 0.45, 0.04), o)
        else:
            body = Geo()
            _car_body(body, st["size"][0], st["size"][1], st["size"][2], CAR_COLORS[st["color"]], 1)
            for vf, m in zip(*_split(body)):
                G(0).add(xf(vf, x, y, 0, st["heading"]), m)
    for i, s in enumerate(sc["signs"]):
        _sign(G(3000 + i), s, 3000 + i)
    return geos


def _split(geo):
    """Geo -> per-material (verts, faces) lists (to re-add with a transform)."""
    out_vf, out_m = [], []
    for k, m in enumerate(geo.mats):
        faces = [f for f, mi in zip(geo.f, geo.mi) if mi == k]
        used = sorted({i for f in faces for i in f})
        remap = {o: n for n, o in enumerate(used)}
        out_vf.append(([geo.v[i] for i in used], [tuple(remap[i] for i in f) for f in faces]))
        out_m.append(m)
    return out_vf, out_m


def _zebra(geo, poly, material, z):
    p = np.asarray(poly, float)
    walk = p[1] - p[0]
    Lw = np.linalg.norm(walk)
    u = walk / Lw
    along = p[3] - p[0]
    for t in np.arange(0.3, Lw - 0.3, 1.0):
        a = p[0] + u * t
        geo.add(g_poly([a, a + u * 0.5, a + u * 0.5 + along, a + along], z), material)


def _arrow(geo, style, pts, material, z):
    a, b = np.asarray(pts[0], float), np.asarray(pts[1], float)
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    n = np.array([-u[1], u[0]])

    def P(s, t):
        return a + u * s + n * t

    polys = []
    if style == "arrow_L":
        polys = [[P(0, -0.15), P(L * 0.55, -0.15), P(L * 0.55, 0.15), P(0, 0.15)],
                 [P(L * 0.4, 0.15), P(L * 0.55, 0.15), P(L * 0.55, 0.9), P(L * 0.4, 0.9)],
                 [P(L * 0.25, 0.9), P(L * 0.7, 0.9), P(L * 0.475, 1.5)]]
    elif style == "arrow_SR":
        polys = [[P(0, -0.15), P(L * 0.6, -0.15), P(L * 0.6, 0.15), P(0, 0.15)],
                 [P(L * 0.6, -0.45), P(L * 0.6, 0.45), P(L, 0.0)],
                 [P(L * 0.25, -0.15), P(L * 0.4, -0.15), P(L * 0.4, -0.9), P(L * 0.25, -0.9)],
                 [P(L * 0.1, -0.9), P(L * 0.55, -0.9), P(L * 0.325, -1.5)]]
    else:   # merge left
        polys = [[P(0, -0.15), P(L * 0.6, 0.7), P(L * 0.6, 1.0), P(0, 0.15)],
                 [P(L * 0.45, 0.4), P(L * 0.75, 1.3), P(L, 0.4)]]
    for q in polys:
        geo.add(g_poly(q, z), material)


def _trees(geo, sc, rng):
    trunk = mat("trunk", (0.18, 0.11, 0.06))
    leaves = [mat(f"leaf{i}", c, rough=0.9) for i, c in enumerate([(0.08, 0.22, 0.05), (0.12, 0.28, 0.07), (0.16, 0.24, 0.06)])]
    sight = np.asarray(sc.get("sightlines", []), float).reshape(-1, 6)
    skipped = 0
    for s in sc["surfaces"]:
        p = np.asarray(s["poly"], float)
        if s["kind"] != "sidewalk" or len(p) != 4:
            continue
        e = [np.linalg.norm(p[(k + 1) % 4] - p[k]) for k in range(4)]
        k = int(np.argmax(e))
        a, b, c = p[k], p[(k + 1) % 4], p[(k + 2) % 4]
        u = (b - a) / e[k]
        w = c - b
        for t in np.arange(rng.uniform(4, 12), e[k], rng.uniform(14, 26)):
            if rng.rand() < 0.5:
                continue
            q = a + u * t + w * 0.75
            h = rng.uniform(3.5, 6.0)
            r = rng.uniform(1.2, 2.0)
            leaf = leaves[rng.randint(3)]
            if crown_blocks((q[0], q[1], h * 0.72), r, sight, margin=0.6):   # drawn but left out; 0.6 m covers a sign face
                skipped += 1
                continue
            geo.add(g_cyl(q[0], q[1], 0.15, h * 0.6, 0.15), trunk)
            geo.add(g_blob(q[0], q[1], h * 0.72, r), leaf)
    print(f"trees skipped: {skipped}", flush=True)


def _sign(geo, s, index):
    x, y = s["pos"]
    h = s["heading"]
    pole = mat("pole", (0.5, 0.5, 0.52), metal=0.6, rough=0.4)
    white = mat("sign_white", (0.9, 0.9, 0.9))
    geo.add(g_cyl(x, y, 0.0, 2.3, 0.04), pole)
    kind = s["kind"]

    def local(vf):
        return xf(vf, x, y, 0.0, h)

    def label(body, size, m, dx=0.045):
        text_obj(f"t{index}", body, size, m, (x + dx * math.cos(h), y + dx * math.sin(h), 2.3), h, index=index)

    if kind == "stop":
        n = 8
        a = np.linspace(0, 2 * math.pi, n, endpoint=False) + math.pi / 8
        geo.add(local(([(0.03, 0.38 * math.cos(t), 2.3 + 0.38 * math.sin(t)) for t in a], [tuple(range(n))])),
                mat("stop_red", (0.6, 0.02, 0.02), rough=0.4))
        label("STOP", 0.2, white)
    elif kind in ("speed", "school_zone"):
        col = (0.9, 0.9, 0.9) if kind == "speed" else (0.85, 0.75, 0.05)
        geo.add(local(g_disc((0.03, 0, 2.3), 0.0, 0.33, 16)), mat("sign_" + kind, col))
        geo.add(local(g_disc((0.025, 0, 2.3), 0.0, 0.38, 16)), mat("sign_ring", (0.6, 0.03, 0.03)))
        label(str(s.get("value") or ""), 0.3, mat("sign_black", (0.03, 0.03, 0.03)))
    elif kind == "lane_ends":
        k = float(s.get("size", 1.0))
        geo.add(local(([(0.03, 0, 2.0), (0.03, 0.4 * k, 2.0 + 0.4 * k), (0.03, 0, 2.0 + 0.8 * k), (0.03, -0.4 * k, 2.0 + 0.4 * k)],
                       [(0, 1, 2, 3)])), mat("sign_yel", (0.85, 0.7, 0.05)))
    elif kind == "crosswalk":
        geo.add(local(g_quad((0.03, 0, 2.3), 0.0, 0.6, 0.6)), mat("sign_blue", (0.05, 0.15, 0.6)))
        geo.add(local(([(0.04, -0.2, 2.1), (0.04, 0.2, 2.1), (0.04, 0, 2.48)], [(0, 1, 2)])), white)


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

def build_heads(sc, geos):
    housing = mat("housing", (0.03, 0.03, 0.03), rough=0.5)
    pole = mat("pole", (0.5, 0.5, 0.52), metal=0.6, rough=0.4)
    cols = dict(red=(1.0, 0.05, 0.02), yellow=(1.0, 0.55, 0.0), green=(0.1, 1.0, 0.3), arrow=(0.1, 1.0, 0.3))
    cols.update({k: tuple(v) for k, v in sc.get("signal_colors", {}).items()})   # per-build lamp colours
    heads = []
    for i, h in enumerate(sc["heads"]):
        x, y, z = h["pos"]
        hd = h["heading"]
        g = geos.setdefault(2000 + i, Geo())

        def local(vf):
            return xf(vf, x, y, z, hd)

        lamps = {}
        if h["kind"] == "veh_pole" and not POLE_HEAD[0]:
            pass     # the pole alone, drawn with the mast arm below
        elif h["kind"] in ("veh", "veh_pole", "veh_near"):
            S = SIG[0]
            names = ["red", "yellow", "green"]
            if _horizontal(h):
                # side by side, red on the driver's left (-y): red, yellow, [arrow], green
                order = ["red", "yellow", "arrow", "green"] if h["arrow"] else ["red", "yellow", "green"]
                n = len(order)
                g.add(local(g_box(-0.12, 0, 0, 0.25, 0.36 * n * S + 0.1 * S, 0.46 * S)), housing)
                where = {nm: ((j - (n - 1) / 2) * 0.36 * S, 0.0) for j, nm in enumerate(order)}
                if h["arrow"]:
                    names.append("arrow")
            else:
                g.add(local(g_box(-0.12, 0, 0, 0.25, 0.36 * S, HOUSING * S)), housing)
                if h["arrow"]:   # the arrow sits beside the green ball, on the driver's left
                    g.add(local(g_box(-0.12, -0.36 * S, -0.36 * S, 0.25, 0.36 * S, 0.46 * S)), housing)
                    names.append("arrow")
                where = {nm: ((-0.36 * S, -0.36 * S) if nm == "arrow" else (0.0, (0.36 - 0.36 * k) * S))
                         for k, nm in enumerate(names)}
            for k, nm in enumerate(names):
                lamps[nm] = lamp_mat(f"head{i}_{nm}", cols[nm])
                ly, lz = where[nm]
                geos.setdefault(5000 + 4 * i + k, Geo()).add(local(g_disc((0.01, ly, lz), 0.0, 0.13 * S)), lamps[nm])
            if h["kind"] in ("veh_pole", "veh_near"):
                g.add(local(g_cyl(-0.15, 0, -z, -HOUSING * S / 2, 0.08)), pole)
        else:
            g.add(local(g_box(-0.1, 0, 0, 0.2, 0.32 * CUE[0], 0.62 * CUE[0])), housing)
            lamps["dont"] = lamp_mat(f"head{i}_dont", (1.0, 0.35, 0.05))
            lamps["walk"] = lamp_mat(f"head{i}_walk", (0.9, 0.95, 1.0))
            g.add(local(g_quad((0.005, 0, 0.15 * CUE[0]), 0.0, 0.24 * CUE[0], 0.24 * CUE[0])), lamps["dont"])
            g.add(local(g_quad((0.005, 0, -0.15 * CUE[0]), 0.0, 0.24 * CUE[0], 0.24 * CUE[0])), lamps["walk"])
            g.add(local(g_cyl(-0.12, 0, -z, -0.3, 0.06)), pole)
            if h["countdown"]:
                lamps["cd"] = lamp_mat(f"head{i}_cd", (1.0, 0.45, 0.05))
                lamps["cd_text"] = text_obj(f"head{i}_cdt", "", 0.22, lamps["cd"],
                                            (x + 0.01 * math.cos(hd), y + 0.01 * math.sin(hd), z - 0.48), hd, index=2000 + i)
        heads.append(dict(spec=h, lamps=lamps))
    # mast poles and arms: one per approach, from the far-right pole over the lanes
    for h in sc["heads"]:
        if h["kind"] != "veh_pole":
            continue
        x, y, _ = h["pos"]
        g0 = geos.setdefault(0, Geo())
        vh = [g for g in sc["heads"] if g["kind"] == "veh" and g["leg"] == h["leg"]]
        top = max(g["pos"][2] + _half_height(g) for g in vh) if vh else 5.75
        arm = top + 0.1
        g0.add(g_cyl(x, y, 0.0, arm + 0.15, 0.12), pole)
        if vh:
            far = max(vh, key=lambda g: (g["pos"][0] - x) ** 2 + (g["pos"][1] - y) ** 2)
            d = np.array(far["pos"][:2]) - np.array([x, y])
            L = np.linalg.norm(d)
            g0.add(xf(g_box(L / 2, 0, arm, L, 0.12, 0.12), x, y, 0, math.atan2(d[1], d[0])), pole)
            for vhh in vh:   # short hanger from the arm to each head
                g0.add(g_cyl(vhh["pos"][0], vhh["pos"][1], vhh["pos"][2] + _half_height(vhh), arm, 0.04), pole)
    return heads


def update_heads(heads, sc, f):
    sig = sc["signals"][f]
    for hd in heads:
        h, L = hd["spec"], hd["lamps"]
        if h["group"] not in sc["groups"]:
            continue
        ball, arrow, ped, cd = sig[sc["groups"].index(h["group"])]
        if not L:
            continue
        if h["kind"] in ("veh", "veh_pole", "veh_near"):
            set_emit(L["red"], ball == 0, LAMP_ON["signal"])
            set_emit(L["yellow"], ball == 1, LAMP_ON["signal"])
            set_emit(L["green"], ball == 2, LAMP_ON["signal"])
            if "arrow" in L:
                set_emit(L["arrow"], arrow > 0, LAMP_ON["signal"])
        else:
            flash_on = (f % 5) < 3
            set_emit(L["walk"], ped == 1, LAMP_ON["ped"])
            set_emit(L["dont"], ped == 0 or (ped == 2 and flash_on), LAMP_ON["ped"])
            if "cd_text" in L:
                body = str(int(cd)) if ped == 2 and cd >= 0 else ""
                if L["cd_text"].data.body != body:
                    L["cd_text"].data.body = body
                set_emit(L["cd"], ped == 2, LAMP_ON["ped"])


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

def _car_body(geo, L, W, H, color, typ):
    paint = mat(f"paint{color}", color, rough=0.35, metal=0.3)
    glass = mat("glass", (0.03, 0.04, 0.06), rough=0.1)
    tire = mat("tire", (0.02, 0.02, 0.02), rough=0.9)
    head = mat("headlight", (0.8, 0.8, 0.75), rough=0.2, emit=(1.0, 0.95, 0.85), strength=0.5)
    r = 0.33 if typ <= 2 else 0.5
    if typ in (1, 2):
        zb, hb = 0.22, 0.45 * H
        hc = H - zb - hb
        geo.add(g_box(0, 0, zb + hb / 2, L, W, hb), paint)
        geo.add(g_box(-0.1 * L, 0, zb + hb + hc / 2, 0.55 * L, W * 0.92, hc, taper=0.35), glass)
        geo.add(g_box(-0.1 * L, 0, H - 0.03, 0.42 * L, W * 0.8, 0.06), paint)
    elif typ == 3:
        cab = 2.0
        geo.add(g_box(L / 2 - cab / 2, 0, 0.35 + H * 0.4, cab, W, H * 0.8), paint)
        geo.add(g_box(L / 2 - 0.02, 0, 0.35 + H * 0.55, 0.06, W * 0.85, H * 0.3), glass)
        geo.add(g_box(-cab / 2, 0, 0.35 + (H - 0.35) / 2 + 0.17, L - cab - 0.2, W, H - 0.35), mat("cargo", (0.8, 0.8, 0.78)))
    else:
        geo.add(g_box(0, 0, 0.3 + (H - 0.3) / 2, L, W, H - 0.3), paint)
        geo.add(g_box(0.2, 0, 0.3 + (H - 0.3) * 0.65, L * 0.92, W + 0.02, (H - 0.3) * 0.32), glass)
        geo.add(g_box(L / 2 + 0.005, 0, 0.3 + (H - 0.3) * 0.55, 0.02, W * 0.85, (H - 0.3) * 0.5), glass)
    for sx in (L / 2 - 0.8 - (0.3 if typ >= 3 else 0.0), -L / 2 + 0.8):
        for sy in (W / 2 - 0.05, -W / 2 + 0.05):
            geo.add(g_box(sx, sy, r, 2 * r, 0.22, 2 * r), tire)
    for sy in (W / 2 - 0.25, -W / 2 + 0.25):
        geo.add(g_quad((L / 2, sy, 0.65 if typ <= 2 else 0.8), 0.0, 0.3, 0.12, offset=0.01), head)


def build_vehicle(slot, typ, L, W, H, color):
    body = Geo()
    _car_body(body, L, W, H, color, typ)
    root = body.obj(f"veh{slot}", index=1 + slot)
    lamps = dict(brake=lamp_mat(f"veh{slot}_brake", (1.0, 0.02, 0.01)),
                 left=lamp_mat(f"veh{slot}_left", (1.0, 0.45, 0.0)),
                 right=lamp_mat(f"veh{slot}_right", (1.0, 0.45, 0.0)))
    zl = 0.75 if typ <= 2 else 1.0
    zf = 0.65 if typ <= 2 else 0.8
    gb = Geo()
    c, ci = CUE[0], IND[0]
    # rear cluster per side: indicator at the outer corner, brake lamp just inboard of it
    ind_w = 0.24 * ci
    br_w = min(0.30 * c, W / 2 - ind_w - 0.07)
    for sgn in (1, -1):
        gb.add(g_quad((-L / 2, sgn * (W / 2 - ind_w - 0.04 - br_w / 2), zl), math.pi, br_w, 0.15 * c, offset=0.012),
               lamps["brake"])
    if typ in (3, 4):   # high centre brake light
        gb.add(g_quad((-L / 2, 0, H - 0.3), math.pi, 0.5 * c, 0.1 * c, offset=0.012), lamps["brake"])
    objs = [gb.obj(f"veh{slot}_brake", index=1000 + 4 * slot, parent=root)]
    for k, (side, sgn) in enumerate((("left", 1), ("right", -1))):
        g = Geo()
        y_out = sgn * (W / 2 - 0.02 - ind_w / 2)
        g.add(g_quad((-L / 2, y_out, zl), math.pi, ind_w, 0.16 * ci, offset=0.013), lamps[side])
        g.add(g_quad((L / 2, y_out, zf), 0.0, ind_w, 0.14 * ci, offset=0.013), lamps[side])
        g.add(g_quad((L / 2 - 0.9, sgn * W / 2, zf + 0.05), sgn * math.pi / 2, 0.2 * ci, 0.08 * ci, offset=0.01), lamps[side])
        objs.append(g.obj(f"veh{slot}_{side}", index=1000 + 4 * slot + 1 + k, parent=root))
    return dict(root=root, lamps=lamps, kind="veh", children=objs)


def build_ped(slot, typ, color):
    s = 1.0 if typ == 5 else 0.68
    shirt = mat(f"shirt{color}", color, rough=0.8)
    pants = mat("pants", (0.08, 0.09, 0.14), rough=0.8)
    skin = mat("skin", (0.62, 0.44, 0.32), rough=0.6)
    body = Geo()
    body.add(g_box(0, 0, 1.2 * s, 0.24 * s, 0.42 * s, 0.6 * s), shirt)
    body.add(g_box(0.02 * s, 0, 1.62 * s, 0.2 * s, 0.18 * s, 0.24 * s), skin)
    root = body.obj(f"ped{slot}", index=1 + slot)
    legs = []
    for sy in (0.1 * s, -0.1 * s):
        g = Geo()
        g.add(g_box(0, 0, -0.45 * s, 0.14 * s, 0.14 * s, 0.9 * s), pants)
        g.add(g_box(0.06 * s, 0, -0.87 * s, 0.26 * s, 0.12 * s, 0.07 * s), pants)
        leg = g.obj(f"ped{slot}_leg{sy:.2f}", index=1 + slot, parent=root)
        leg.location = (0, sy, 0.9 * s)
        legs.append(leg)
    return dict(root=root, legs=legs, kind="ped", children=legs)


def set_visible(ob, vis):
    if ob["root"].hide_render == (not vis):
        return
    ob["root"].hide_render = not vis
    for ch in ob["children"]:
        if ch is not None:
            ch.hide_render = not vis


# ---------------------------------------------------------------------------
# world, camera, compositor
# ---------------------------------------------------------------------------

def setup_world(sc, gpu, samples, tmpdir):
    scn = bpy.context.scene
    look = sc["look"]
    scn.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    cuda = [d for d in prefs.devices if d.type == "CUDA"]
    for d in prefs.devices:
        d.use = False
    cuda[gpu % len(cuda)].use = True
    scn.cycles.device = "GPU"
    scn.cycles.samples = samples
    scn.cycles.use_denoising = True
    scn.cycles.denoiser = "OPENIMAGEDENOISE"
    scn.cycles.denoising_use_gpu = os.environ.get("DT_DENOISE_GPU", "1") == "1"
    scn.render.threads_mode = "FIXED"
    scn.render.threads = int(os.environ.get("DT_BLENDER_THREADS", "8"))
    scn.cycles.max_bounces = 3
    scn.cycles.diffuse_bounces = 2
    scn.cycles.glossy_bounces = 1
    scn.cycles.transmission_bounces = 1
    scn.cycles.use_adaptive_sampling = False
    scn.render.use_persistent_data = True
    scn.render.resolution_x = scn.render.resolution_y = sc["img"]
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    view, look_name = sc.get("view", ["AgX", "AgX - Medium High Contrast"])
    scn.view_settings.view_transform = view
    scn.view_settings.look = look_name

    world = bpy.data.worlds.new("world")
    scn.world = world
    world.use_nodes = True
    nt = world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(look["sun_elevation"])
    sky.sun_rotation = math.radians(look["sun_azimuth"])
    sky.sun_disc = False
    sky.dust_density = 1.0 + 4 * look["fog"]
    nt.links.new(sky.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = 0.35 if look["time"] == "day" else 0.25
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.5 if look["time"] == "day" else 1.8
    sun.color = SUN_COLOR[look["time"]]
    sun.angle = math.radians(1.0)
    so = bpy.data.objects.new("sun", sun)
    scn.collection.objects.link(so)
    el, az = math.radians(look["sun_elevation"]), math.radians(look["sun_azimuth"])
    so.rotation_euler = (math.pi / 2 - el, 0, az + math.pi / 2)

    vl = scn.view_layers[0]
    vl.use_pass_object_index = True
    vl.use_pass_mist = look["fog"] > 0
    world.mist_settings.start = 5.0
    world.mist_settings.depth = 120.0
    world.mist_settings.falloff = "LINEAR"

    scn.use_nodes = True
    ct = scn.node_tree
    for n in list(ct.nodes):
        ct.nodes.remove(n)
    rl = ct.nodes.new("CompositorNodeRLayers")
    comp = ct.nodes.new("CompositorNodeComposite")
    if look["fog"] > 0:
        mul = ct.nodes.new("CompositorNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = look["fog"]
        mix = ct.nodes.new("CompositorNodeMixRGB")
        mix.inputs[2].default_value = (0.62, 0.65, 0.68, 1.0) if look["time"] == "day" else (0.55, 0.45, 0.42, 1.0)
        ct.links.new(rl.outputs["Mist"], mul.inputs[0])
        ct.links.new(mul.outputs[0], mix.inputs[0])
        ct.links.new(rl.outputs["Image"], mix.inputs[1])
        ct.links.new(mix.outputs[0], comp.inputs["Image"])
    else:
        ct.links.new(rl.outputs["Image"], comp.inputs["Image"])
    fo = ct.nodes.new("CompositorNodeOutputFile")
    fo.base_path = tmpdir
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.format.exr_codec = "ZIP"
    fo.file_slots[0].path = "index_"
    ct.links.new(rl.outputs["IndexOB"], fo.inputs[0])

    cam = bpy.data.cameras.new("cam")
    cam.lens_unit = "FOV"
    cam.sensor_fit = "HORIZONTAL"
    cam.angle = math.radians(sc["camera"]["fov"])
    cam.clip_start = 0.3
    cam.clip_end = 400.0
    co = bpy.data.objects.new("cam", cam)
    scn.collection.objects.link(co)
    scn.camera = co
    return co


def place_camera(co, sc, f):
    x, y, h, _ = sc["ego"][f]
    c = sc["camera"]
    co.location = (x + c["forward"] * math.cos(h), y + c["forward"] * math.sin(h), c["height"])
    co.rotation_euler = (math.radians(90 + c["pitch"]), 0.0, h - math.pi / 2)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def render_job(scene_path, out_path, gpu, samples, keep_png=False):
    with open(scene_path) as fh:
        sc = json.load(fh)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _mats.clear()
    CUE[0] = float(sc.get("cue_scale", 1.0))
    SIG[0] = float(sc.get("signal_scale", CUE[0]))
    LAYOUT[0] = sc.get("signal_layout", "vertical")
    POLE_HEAD[0] = bool(sc.get("pole_head", True))
    IND[0] = float(sc.get("indicator_scale", CUE[0]))
    LAMP_ON.update(sc.get("lamp_emit", {}))
    rng = np.random.RandomState(sc["look"]["building_seed"])
    tmpdir = tempfile.mkdtemp(prefix="dtr_", dir="/dev/shm")
    co = setup_world(sc, gpu, samples, tmpdir)
    geos = build_static(sc, rng)
    heads = build_heads(sc, geos)
    for index, g in geos.items():
        g.obj(f"static{index}", index=index)
    agents = np.asarray(sc["agents"], float)       # (T, A, 10)
    lamps = np.asarray(sc["lamps"], int)
    T, A = agents.shape[:2]
    objs = {}
    for a in range(1, A):                            # slot 0 is the ego: the camera rides it
        valid = agents[:, a, 9] > 0
        if not valid.any():
            continue
        f0 = int(np.argmax(valid))
        typ = int(agents[f0, a, 0])
        col = int(sc["colors"].get(str(a), a % 12))
        if typ in VEH_TYPES:
            objs[a] = build_vehicle(a, typ, agents[f0, a, 6], agents[f0, a, 7], agents[f0, a, 8], CAR_COLORS[col])
        else:
            objs[a] = build_ped(a, typ, SHIRT_COLORS[col])
    scn = bpy.context.scene
    S = sc["img"]
    video = np.zeros((T, S, S, 3), np.uint8)
    index = np.zeros((T, S, S), np.int16)
    for f in range(T):
        scn.frame_current = f
        for a, ob in objs.items():
            st = agents[f, a]
            vis = st[9] > 0
            set_visible(ob, vis)
            if not vis:
                continue
            ob["root"].location = (st[1], st[2], 0.0)
            ob["root"].rotation_euler = (0, 0, st[3])
            if ob["kind"] == "veh":
                set_emit(ob["lamps"]["brake"], lamps[f, a, 0], LAMP_ON["brake"])
                set_emit(ob["lamps"]["left"], lamps[f, a, 1], LAMP_ON["left"])
                set_emit(ob["lamps"]["right"], lamps[f, a, 2], LAMP_ON["right"])
            else:
                swing = 0.45 * math.sin(f * 2.4 + a) if st[4] > 0.2 else 0.0
                ob["legs"][0].rotation_euler = (0, swing, 0)
                ob["legs"][1].rotation_euler = (0, -swing, 0)
        update_heads(heads, sc, f)
        place_camera(co, sc, f)
        scn.render.filepath = os.path.join(tmpdir, f"rgb_{f:03d}.png")
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(scn.render.filepath)
        px = np.empty(S * S * 4, np.float32)
        img.pixels.foreach_get(px)
        video[f] = np.clip(px.reshape(S, S, 4)[::-1, :, :3] * 255 + 0.5, 0, 255).astype(np.uint8)
        bpy.data.images.remove(img)
        ex = bpy.data.images.load(os.path.join(tmpdir, f"index_{f:04d}.exr"))
        ix = np.empty(S * S * 4, np.float32)
        ex.pixels.foreach_get(ix)
        index[f] = np.round(ix.reshape(S, S, 4)[::-1, :, 0]).astype(np.int16)
        bpy.data.images.remove(ex)
    np.savez_compressed(out_path, video=video, index=index)
    for fn in os.listdir(tmpdir):
        if keep_png and fn.startswith("rgb_"):
            continue
        os.remove(os.path.join(tmpdir, fn))
    if not keep_png:
        os.rmdir(tmpdir)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--keep-png", action="store_true")
    args = ap.parse_args(argv)
    with open(args.jobs) as fh:
        jobs = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    for scene_path, out_path in jobs:
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        render_job(scene_path, out_path, args.gpu, args.samples, args.keep_png)
        print(f"[render_clip] {out_path} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
