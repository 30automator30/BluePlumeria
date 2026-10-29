#!/usr/bin/env python3
"""ʻŌhiʻa lehua bonsai (resin / MSLA, Saturn 4 Ultra), about 120 mm tall in
its pot. Two parts, both printed straight on the plate:

  tree   (~105 mm) a soil plug with a lump of lava rock, a gnarled trunk of
         fused stems twisting round an S-curved line, flared roots gripping
         the soil and the rock, and an aerial root dropping from the low
         branch. Four branches and a leader end in lehua blossoms: round
         brushes of stamens, each rising from a cluster of little flower
         cups on a knob, over collars of small paired leaves. Side sprigs
         carry more leaf pairs, red leaf buds (liko) and bud clusters.
         Every stamen rises from its flower and curls gently upward, so
         the blossoms need no supports; anything that would still start in
         mid-air gets a slim stem (the same planner as the other pieces).
  pot    an oval bonsai pot with a lip, four feet and two drain holes. A
         45-degree ledge inside holds the soil plug, 0.3 mm clearance.

Usage
  python3 ohia_bonsai.py --voxel 0.2 --preview          # quick look
  python3 ohia_bonsai.py                                # final STLs
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("fractal-plumeria", "plumeria-ring", "hibiscus-bonsai"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import lattice_plumeria as lp  # noqa: E402
from bonsai_hibiscus import box, clear_stems, frame, local, unit  # noqa: E402
from fractal_plumeria import capsule_chain, mesh_volume  # noqa: E402
from lattice_plumeria import BASE_CUT, add, bezier, soft_band  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))
UP = np.array([0, 0, 1.0])

# ---------------------------------------------------------------- layout
SOIL_Z = 4.5                         # soil plug thickness
PLUG = (29.0, 20.0)                  # plug half-axes (oval)
POM_R = 12.5                         # blossom radius: ~25 mm brushes
STAMEN_R = (0.44, 0.36, 0.52)        # filament radius at base, at tip; anther
STAMEN_EL = np.radians(22)           # no stamen leaves its flower flatter than this
LEAF_L, LEAF_W = 7.0, 2.7            # small elliptic leaves (half-width)

POT_H = 20.0
POT_R = (39.0, 29.5)                 # outer half-axes at the rim
FIT = 0.3
LEDGE_Z = POT_H - 2.0 - SOIL_Z       # the plug rests here; soil sits 2 mm below the rim


def oval(x, y, a, b):
    """Signed distance (approximate) to an ellipse of half-axes a, b."""
    return (np.sqrt((x / a) ** 2 + (y / b) ** 2) - 1) * np.minimum(a, b)


# ---------------------------------------------------------------- leaves

def leaf_sdf(fr, L=LEAF_L, W=LEAF_W, T=0.9):
    """Small elliptic ʻōhiʻa leaf with a rounded tip, cupped upward, a
    raised midrib and a short petiole running into the twig."""
    def f(x, y, z):
        u, v, w = local(fr, x, y, z)
        t = u / L
        tc = np.clip(t, 0, 1)
        half = W * np.sin(np.pi * tc ** 0.85) ** 0.7 + 1e-3
        o = np.maximum(np.abs(v) - half, np.maximum(-u, u - L))
        h = 0.10 * v ** 2 + 0.9 * tc ** 2                   # cupped, tip lifts
        d = soft_band(o, np.abs(w - h) - T / 2, 2.0, T / 2)
        rib = np.maximum(np.abs(v) - 0.38, o + 0.8)
        d = smin(d, soft_band(rib, np.abs(w - h - 0.4) - 0.3, 0.38, 0.3), 0.25)
        stalk = capsule_chain(u, v, w, np.array([[-2.6, 0, 0.0], [0.4, 0, 0.0]]),
                              np.array([0.6, 0.5]))
        return smin(d, stalk, 0.4)
    return f, box(fr, (-3.4, -W - 1, -2.5), (L + 1, W + 1, 3.5))


def liko_sdf(base, axis, L=4.6):
    """Leaf bud (liko): a pair of young leaves still pressed together,
    a slim upright spearhead with a seam down each side."""
    fr = frame(base, axis, UP if abs(axis[2]) < 0.95 else np.array([1.0, 0, 0]))

    def f(x, y, z):
        u, v, w = local(fr, x, y, z)
        t = np.clip(u / L, 0, 1)
        r = 1.15 * np.sin(np.pi * np.clip(0.15 + 0.85 * t, 0, 1)) ** 0.8 + 0.25
        body = np.maximum(np.sqrt(v ** 2 + (w / 0.62) ** 2) * 0.8 - r * 0.8, np.maximum(-u, u - L))
        seam = np.abs(w) - 0.12
        return np.maximum(body, -np.maximum(seam, -(np.abs(v) - 0.75 * r)))
    return f, box(fr, (-1, -2, -2), (L + 1, 2, 2))


# ------------------------------------------------------------- blossoms

def pom_stamens(c, axis, R=POM_R, n_fl=16, per=10, seed=0):
    """The stamens of one lehua blossom (an umbel of small flowers on a knob,
    each a cup with a fan of long stamens). Directions spread over the upper
    dome round `axis`; every filament leaves its cup at least STAMEN_EL above
    horizontal and curls upward, so its anther is its highest point."""
    rng = np.random.default_rng(seed)
    fr = frame(c, axis, np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0]))
    _, e3, e1, e2 = fr                                    # e3 along the axis
    cups, out = [], []
    for i in range(n_fl):
        # flower directions: a golden spiral over the knob's upper dome
        h = 1 - (i + 0.5) / n_fl * 1.05                   # cos of angle from axis
        h = np.clip(h, -0.05, 1)
        a = i * GOLDEN_ANGLE + seed
        n = h * e3 + np.sqrt(1 - h * h) * (np.cos(a) * e1 + np.sin(a) * e2)
        cup = c + 2.6 * n
        cups.append((cup, n))
        for k in range(per):
            # fan round the flower's direction, wider for flowers near the top
            g = k * GOLDEN_ANGLE + a
            sp = 0.55 * np.sqrt((k + 0.5) / per)
            q = unit(n + sp * (np.cos(g) * unit(np.cross(n, e3 + 1e-3 * e1))
                               + np.sin(g) * unit(np.cross(n, np.cross(n, e3 + 1e-3 * e1)))))
            el = np.arcsin(np.clip(q[2], -1, 1))
            if el < STAMEN_EL:                            # lift to the minimum
                hz = unit(q * [1, 1, 0] + 1e-6)
                q = np.cos(STAMEN_EL) * hz + np.sin(STAMEN_EL) * UP
            Ls = (R - 3.0) * (0.82 + 0.18 * rng.random())
            p0 = cup + 0.4 * q
            curl = 0.16 * Ls * np.clip(1 - q[2], 0, 1) * UP   # flatter ones curl up more
            p3 = p0 + Ls * q + curl
            pts = bezier(p0, p0 + 0.35 * Ls * q, p0 + 0.7 * Ls * q + 0.4 * curl, p3, n=7)
            out.append(pts)
    return cups, out


def stamen_sdf(pts):
    """Filament tapering to the tip, then a slim club-shaped anther that
    widens gradually (no bead to start in mid-air)."""
    r0, r1, ra = STAMEN_R
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    s = L / L[-1]
    rad = np.where(s < 0.82, r0 + (r1 - r0) * s / 0.82, r1 + (ra - r1) * np.clip((s - 0.82) / 0.12, 0, 1))

    def f(x, y, z):
        return capsule_chain(x, y, z, pts, rad)
    return f, (pts.min(0) - 1.0, pts.max(0) + 1.0)


def knob_sdf(c, axis, cups):
    """The blossom's knob: a rounded receptacle with a little cup for each
    flower, tapering down into the twig."""
    def f(x, y, z):
        q = np.stack([x - c[0], y - c[1], z - c[2]], -1)
        h = q @ axis
        rr = np.sqrt(np.maximum((q * q).sum(-1) - h * h, 0))
        d = np.sqrt(rr ** 2 + (np.minimum(h, 0) / 1.6) ** 2 + np.maximum(h, 0) ** 2) - 2.9
        for p, n in cups:
            g = np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2) - 0.95
            d = smin(d, g, 0.6)
        return d
    return f, (c - 7.0, c + 7.0)


def bud_cluster(c, axis, n=7, seed=1):
    """A cluster of round flower buds on short stalks, like a blossom
    before it opens. Each bud is a teardrop that widens along its stalk."""
    rng = np.random.default_rng(seed)
    tubes = []
    fr = frame(c, axis, np.array([1.0, 0, 0]))
    _, e3, e1, e2 = fr
    for i in range(n):
        h = 1 - (i + 0.5) / n * 0.9
        a = i * GOLDEN_ANGLE + seed
        d = unit(h * e3 + np.sqrt(1 - h * h) * (np.cos(a) * e1 + np.sin(a) * e2) + 0.5 * UP)
        L = 2.6 + 0.8 * rng.random()
        pts = np.array([c + (L * s) * d for s in np.linspace(0, 1, 6)])
        rad = np.array([0.55, 0.6, 0.75, 1.05, 1.25, 0.95])
        tubes.append((pts, rad))
    return tubes


# ------------------------------------------------------------------ tree

def trunk_line():
    """Informal-upright line: leans left, swings back right, then the leader."""
    a = bezier(np.array([0, 0, SOIL_Z - 1.0]), np.array([-1.0, 0.5, 16.0]),
               np.array([-9.0, 2.0, 25.0]), np.array([-6.5, 1.5, 38.0]), n=80)
    t = a[-1] - a[-2]
    b = bezier(a[-1], a[-1] + 10.0 * unit(t), np.array([4.0, -1.0, 56.0]),
               np.array([2.0, 0.0, 68.0]), n=70)
    c = bezier(b[-1], b[-1] + 6.0 * unit(b[-1] - b[-2]), np.array([-2.0, 2.0, 78.0]),
               np.array([-1.0, 2.5, 87.0]), n=40)
    return np.vstack([a, b[1:], c[1:]])


def trunk_tubes(tubes):
    """Five stems fused into one gnarled trunk, twisting half a turn round
    the line, so the bark reads as deep twisted ridges. They taper to the
    leader and flare into buttress roots at the soil."""
    line = trunk_line()
    z = line[:, 2]
    s = np.linspace(0, 1, len(line))
    R = np.interp(z, [SOIL_Z, 20, 45, 68, 87], [5.6, 4.6, 3.6, 2.6, 1.3])
    flare = 6.0 * np.exp(-(z - SOIL_Z) / 5.0)
    tan = np.gradient(line, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    e1 = np.cross(tan, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(tan, e1)
    for j in range(5):
        th = 2 * np.pi * j / 5 + 1.1 * np.pi * s + 0.25 * np.sin(9 * s + j)
        off = (0.72 * R + flare) * (1 + 0.15 * np.sin(5 * s * np.pi + 2 * j))
        path = line + off[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        r = 0.5 * R * (1 + 0.15 * np.sin(7 * s + j)) + 0.25 * flare
        # each stem runs on as a root, out and down into the soil
        p0, d0 = path[0], unit(path[0] - path[3])
        out = unit([p0[0] - line[0][0], p0[1] - line[0][1], 0])
        reach = 8.0 + 3.0 * (j % 2)
        tip = p0 + reach * out + np.array([0, 0, SOIL_Z - 1.6 - p0[2]])
        root = bezier(p0, p0 + 2.0 * d0, tip - 3.0 * out + [0, 0, 1.2], tip, n=20)[::-1]
        tubes.append((np.vstack([root[:-1], path]),
                      np.concatenate([np.linspace(1.0, r[0], len(root) - 1), r])))
    tubes.append((line, 0.55 * R))                         # the core between the stems
    return line


ROCK_C = np.array([14.0, -8.0, SOIL_Z - 0.5])


def rock_sdf(x, y, z):
    """A lump of vesicular lava rock half sunk in the soil."""
    q = np.stack([x - ROCK_C[0], y - ROCK_C[1], z - ROCK_C[2]], -1)
    lump = np.sqrt((q[..., 0] / 9.0) ** 2 + (q[..., 1] / 6.5) ** 2 + (q[..., 2] / 6.0) ** 2) - 1
    lump = lump * 6.0 + 0.6 * np.sin(1.3 * x + 0.5) * np.sin(1.1 * y) * np.sin(1.7 * z + 1) \
        + 0.35 * np.sin(2.9 * x + 2.0 * y) * np.sin(2.3 * z)
    d = lump
    rng = np.random.default_rng(7)
    for _ in range(14):                                   # vesicles
        a, e = rng.uniform(0, 2 * np.pi), rng.uniform(0.1, 1.2)
        p = ROCK_C + np.array([9.0 * np.cos(a) * np.cos(e), 6.5 * np.sin(a) * np.cos(e), 6.0 * np.sin(e)])
        d = np.maximum(d, -(np.sqrt(((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2)) - rng.uniform(0.5, 1.0)))
    return d


def at_height(line, z):
    i = np.argmin(np.abs(line[:, 2] - z))
    return line[i]


def tree_parts():
    """Trunk, branches, twigs, and where blossoms, buds, leaves and liko go."""
    tubes, things = [], []
    line = trunk_tubes(tubes)

    # main branches: (start height on the trunk, end point, start radius)
    branches = [
        (33.0, np.array([-33.0, -6.0, 50.0]), 2.6),      # long low branch, left
        (47.0, np.array([29.0, -6.0, 60.0]), 2.3),       # right
        (55.0, np.array([5.0, 25.0, 68.0]), 2.0),        # back
        (62.0, np.array([-9.0, -24.0, 70.0]), 1.9),      # front
    ]
    ends = []
    for z0, p3, r0 in branches:
        p0 = at_height(line, z0)
        d = unit(p3 - p0)
        out = unit(d * [1, 1, 0])
        br = bezier(p0, p0 + 0.35 * np.linalg.norm(p3 - p0) * unit(out + 0.9 * UP),
                    p3 - 6.0 * unit(out + 0.25 * UP), p3, n=50)
        tubes.append((br, np.linspace(r0, 1.25, 50)))
        ends.append((br, out))
    # leader top
    ends.append((line[-30:], unit(line[-1] - line[-8]) * [1, 1, 0] + 1e-3))

    # aerial root: drops from under the low branch into the soil
    br0 = ends[0][0]
    hang = br0[int(0.62 * len(br0))] - [0, 0, 1.0]
    foot = np.array([hang[0] + 1.5, hang[1] + 1.0, SOIL_Z - 1.5])
    ar = bezier(foot, foot + [0.5, 0, 0.4 * (hang[2] - foot[2])],
                hang - [1.0, 0.5, 0.35 * (hang[2] - foot[2])], hang, n=40)
    ar = ar + np.column_stack([0.6 * np.sin(np.linspace(0, 5, 40)), 0 * ar[:, 0], 0 * ar[:, 0]]) \
        * np.sin(np.linspace(0, np.pi, 40))[:, None]
    tubes.append((ar, np.linspace(1.3, 0.75, 40)))
    # a surface root over the lava rock
    rr = bezier(np.array([3.5, -2.0, SOIL_Z + 2.5]), np.array([8.0, -5.0, SOIL_Z + 4.8]),
                np.array([15.0, -9.0, SOIL_Z + 4.0]), np.array([20.0, -11.0, SOIL_Z - 1.0]), n=30)
    tubes.append((rr, np.linspace(1.8, 0.8, 30)))

    poms = []
    for j, (br, out) in enumerate(ends):
        tip = br[-1]
        side = unit(np.cross(UP, out))
        # the blossom: a twig rising out of the branch end, standing clear
        # of a cushion of leafy sprigs
        axis = unit(0.45 * out + UP)
        c = tip + 8.5 * axis + 0.5 * out
        tw = bezier(tip, tip + 2.5 * unit(br[-1] - br[-3]), c - 3.5 * axis, c - 1.0 * axis, n=16)
        tubes.append((tw, np.linspace(1.25, 1.0, 16)))
        poms.append((c, axis, j))
        a_out = np.arctan2(out[1], out[0])
        roots = (br[-1], br[-6]) if j < 4 else (line[-1], line[-6])
        fan = (-2.0, -1.05, -0.2, 0.7, 1.5, 2.6) if j < 4 else (0.3, 1.35, 2.4, 3.45, 4.5, 5.55)
        for m, da in enumerate(fan):
            a = a_out + da + 0.15 * np.sin(3 * j + m)
            sd = unit(np.cos(np.radians(24)) * np.array([np.cos(a), np.sin(a), 0])
                      + np.sin(np.radians(24)) * UP)
            p = roots[m % 2]
            L = 10.5 + 1.5 * np.sin(2 * j + m)
            sp = bezier(p, p + 2.5 * unit(sd + 0.6 * UP), p + L * sd - 3.0 * sd, p + L * sd, n=18)
            tubes.append((sp, np.linspace(1.0, 0.75, 18)))
            sprig_leaves(things, sp, sd, j + m, nodes=(0.3, 0.55, 0.8))
            end = sp[-1]
            if (j + m) % 4 == 1:
                tubes += bud_cluster(end + 0.4 * sd, unit(sd + UP), seed=j * 7 + m)
            else:
                things.append(("liko", (end - 0.3 * sd, unit(0.35 * sd + UP))))
        # sprigs further back along the branch
        if j < 4:
            for m, (f_, sgn) in enumerate(((0.5, 1), (0.72, -1))):
                p = br[int(f_ * (len(br) - 1))]
                sd = unit(0.8 * sgn * side + 0.4 * out + 0.55 * UP)
                L = 11.0 - 1.5 * m
                sp = bezier(p, p + 3.0 * unit(sd + UP), p + L * sd - 3.0 * sd, p + L * sd, n=18)
                tubes.append((sp, np.linspace(1.1, 0.8, 18)))
                sprig_leaves(things, sp, sd, j + m, nodes=(0.3, 0.55, 0.8))
                end = sp[-1]
                if (j + m) % 2 == 0:
                    tubes += bud_cluster(end + 0.6 * sd, unit(sd + UP), seed=j * 3 + m)
                else:
                    things.append(("liko", (end - 0.3 * sd, unit(0.3 * sd + UP))))
    # leaf pairs along the branches themselves
    for j, (br, out) in enumerate(ends[:4]):
        sprig_leaves(things, br[int(0.4 * len(br)):], unit(br[-1] - br[int(0.4 * len(br))]), j + 7,
                     nodes=(0.3, 0.6))

    kept = [t for t in things if not clips_pom(t, poms)]
    return tubes, kept, poms


def sprig_leaves(things, sp, sd, seed, nodes=(0.35, 0.7)):
    """Opposite leaf pairs along a sprig, each pair turned 90 degrees from
    the last, leaves angled forward and up."""
    side = unit(np.cross(UP, sd * [1, 1, 0] + 1e-6))
    w = unit(np.cross(sd, side))
    for k, f_ in enumerate(nodes):
        q = sp[int(f_ * (len(sp) - 1))]
        ax = side if k % 2 == 0 else w
        for sgn in (1, -1):
            ld = unit(0.75 * sgn * ax + 0.55 * sd + 0.45 * UP)
            if ld[2] < 0.42:                              # keep them rising
                ld = unit(ld * [1, 1, 0] * 0.9 / max(np.linalg.norm(ld[:2]), 1e-6) + 0.45 * UP)
            things.append(("leaf", frame(q + 1.4 * ld, ld)))


def clips_pom(thing, poms):
    """True for a leaf or liko that would poke into a blossom's stamens."""
    if thing[0] == "leaf":
        base, u, _, _ = thing[1]
        pts = base + np.linspace(0, 1.0, 8)[:, None] * LEAF_L * u
    else:
        base, u = thing[1]
        pts = base + np.linspace(0, 1.0, 5)[:, None] * 4.6 * u
    for c, axis, _ in poms:
        q = pts - c
        horiz = np.linalg.norm(q[:, :2], axis=1)
        inside = (np.linalg.norm(q, axis=1) < POM_R + 1.0) & (q[:, 2] > np.tan(STAMEN_EL - 0.2) * horiz - 2.5)
        if np.any(inside):
            return True
    return False


def bounds():
    return np.array([-50.0, -42.0, -0.6]), np.array([50.0, 42.0, 112.0])


def add_tube(vol, xs, ys, zs, pts, rr, k):
    """Blend a whole tube in at once. Its pieces are joined with a plain
    min first; blending them one by one would bulge at every joint."""
    pad = rr.max() + 1.0
    sl = lp.sub(xs, ys, zs, pts.min(0) - pad, pts.max(0) + pad)
    if any(s_.stop <= s_.start for s_ in sl):
        return
    sx, sy, sz = xs[sl[0]], ys[sl[1]], zs[sl[2]]
    buf = np.full((len(sx), len(sy), len(sz)), 3.0, np.float32)
    for k0 in range(0, len(pts) - 1, 10):
        seg, rad = pts[k0:k0 + 11], rr[k0:k0 + 11]
        p_ = rad.max() + 1.0
        add(buf, sx, sy, sz, seg.min(0) - p_, seg.max(0) + p_,
            lambda x, y, z, s_=seg, r_=rad: capsule_chain(x, y, z, s_, r_), k=0)
    vol[sl] = smin(vol[sl], buf, k)


def build_tree(voxel, stems=(), quiet=False, region=None):
    say = (lambda *a: None) if quiet else print
    t0 = time.time()
    tubes, things, poms = tree_parts()
    lo, hi = bounds() if region is None else (np.asarray(region[0]), np.asarray(region[1]))
    xs = np.arange(lo[0], hi[0], voxel, dtype=np.float32)
    ys = np.arange(lo[1], hi[1], voxel, dtype=np.float32)
    zs = np.arange(lo[2], hi[2], voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)

    def plug(x, y, z):
        bumps = (0.22 * np.sin(1.3 * x + 0.9 * y) * np.sin(-0.7 * x + 1.6 * y + 1.0)
                 + 0.14 * np.sin(2.9 * x - 1.1 * y + 2.0) * np.sin(0.8 * x + 2.6 * y)
                 + 0.08 * np.sin(4.7 * x + 3.3 * y) * np.sin(-3.9 * x + 4.1 * y))
        return np.maximum(oval(x, y, *PLUG), np.maximum(-z, z - SOIL_Z - bumps))
    add(vol, xs, ys, zs, (-PLUG[0] - 1, -PLUG[1] - 1, -1), (PLUG[0] + 1, PLUG[1] + 1, SOIL_Z + 1),
        plug, k=0)
    add(vol, xs, ys, zs, ROCK_C - [11, 9, 7], ROCK_C + [11, 9, 8], rock_sdf, k=0.8)

    for pts, rr in tubes:
        add_tube(vol, xs, ys, zs, pts, np.broadcast_to(rr, (len(pts),)), k=0.8)
    say(f"  trunk and branches {time.time() - t0:.0f}s")
    for kind, spec in things:
        f, (blo, bhi) = leaf_sdf(spec) if kind == "leaf" else liko_sdf(*spec)
        add(vol, xs, ys, zs, blo, bhi, f, k=0.4, chunk=8)
    say(f"  {len(things)} leaves and liko {time.time() - t0:.0f}s")
    n_st = 0
    for c, axis, j in poms:
        cups, sts = pom_stamens(c, axis, seed=j)
        f, (blo, bhi) = knob_sdf(c, axis, cups)
        add(vol, xs, ys, zs, blo, bhi, f, k=0.6, chunk=8)
        for pts in sts:
            f, (blo, bhi) = stamen_sdf(pts)
            add(vol, xs, ys, zs, blo, bhi, f, k=0.3, chunk=16)
        n_st += len(sts)
    say(f"  {len(poms)} blossoms, {n_st} stamens {time.time() - t0:.0f}s")
    for pts, rr in stems:
        add(vol, xs, ys, zs, pts.min(0) - 2.5, pts.max(0) + 2.5,
            lambda x, y, z, s_=pts, r_=rr: capsule_chain(x, y, z, s_, r_), k=0.8)
    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])
    return vol, (xs[0], ys[0], zs[0])


def plan_stems(voxel=0.3, passes=8):
    """Slim stems up to anything that would start in mid-air (coarse grid)."""
    stems = []
    for i in range(passes):
        vol, origin = build_tree(voxel, stems, quiet=True)
        isl = lp.find_islands(vol, voxel, origin)
        print(f"  stem pass {i + 1}: {len(isl)} island(s), {len(stems)} stem(s)")
        if not isl:
            break
        stems += clear_stems(isl, vol, voxel, origin)
    return stems


# ------------------------------------------------------------------- pot

def build_pot(voxel):
    a0, b0 = POT_R
    xs = np.arange(-a0 - 3, a0 + 3, voxel, dtype=np.float32)
    ys = np.arange(-b0 - 3, b0 + 3, voxel, dtype=np.float32)
    zs = np.arange(-0.6, POT_H + 1.5, voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)

    def pot(x, y, z):
        k = z / POT_H
        lip = 1.2 * np.clip((z - (POT_H - 3.7)) / 1.2, 0, 1)        # 45-degree underside
        da = -2.4 * (1 - k) + lip                                   # tapers toward the foot
        ang = np.arctan2(y / b0, x / a0)
        feet = np.cos(4 * (ang - np.pi / 4)) > 0.82                 # four feet on the diagonals
        inset = np.where(feet, 0.0, 1.0 * np.clip(3.5 - z, 0, 1))
        outer = np.maximum(oval(x, y, a0 + da - inset - 1.2, b0 + da - inset - 1.2),
                           np.maximum(-z, z - POT_H))
        seat = np.maximum(oval(x, y, PLUG[0] + FIT, PLUG[1] + FIT), LEDGE_Z - z)
        shrink = 2.0 * np.clip((LEDGE_Z - z) / 2.0, 0, 1)
        hollow = np.maximum(oval(x, y, PLUG[0] + FIT - shrink, PLUG[1] + FIT - shrink),
                            np.maximum(2.5 - z, z - LEDGE_Z))
        drains = np.minimum(np.hypot(x - 12, y), np.hypot(x + 12, y)) - 2.5
        cut = np.minimum(np.minimum(seat, hollow), np.maximum(drains, z - 3.0))
        return np.maximum(outer, -cut)
    add(vol, xs, ys, zs, (-a0 - 3, -b0 - 3, -1), (a0 + 3, b0 + 3, POT_H + 1), pot, k=0)
    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])
    return vol, (xs[0], ys[0], zs[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.12, help="mesh resolution, mm")
    ap.add_argument("--preview", action="store_true", help="skip support planning")
    ap.add_argument("--part", choices=["tree", "pot", "both"], default="both")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    for part in (["tree", "pot"] if args.part == "both" else [args.part]):
        t0 = time.time()
        if part == "tree":
            vol, origin = build_tree(args.voxel, () if args.preview else plan_stems())
        else:
            vol, origin = build_pot(args.voxel)
        mesh = decimate(mesh_volume(vol, args.voxel, origin), 0.006)
        print(f"{part}: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
              f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, "
              f"{time.time() - t0:.0f}s")
        out = os.path.join(args.outdir, f"ohia_bonsai_{part}.stl")
        mesh.export(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
