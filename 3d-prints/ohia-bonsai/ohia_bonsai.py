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
STAMEN_R = (0.40, 0.34, 0.58)        # filament radius at base, at tip; anther bead
STAMEN_EL = np.radians(25)           # no stamen leaves its flower flatter than this
LEAF_L, LEAF_W = 7.5, 3.3
BIRD_BRANCH = 1                      # the ʻiʻiwi perches on the right branch
BIRD_S = 1.3                         # ...about 33 mm from bill tip to tail
BRANCH_R = (2.6, 2.3, 2.0, 1.9)      # branch radius where it leaves the trunk            # small elliptic leaves (half-width)

POT_H = 20.0
POT_R = (39.0, 29.5)                 # outer half-axes at the rim
FIT = 0.3
LEDGE_Z = POT_H - 2.0 - SOIL_Z       # the plug rests here; soil sits 2 mm below the rim


def oval(x, y, a, b):
    """Signed distance (approximate) to an ellipse of half-axes a, b."""
    return (np.sqrt((x / a) ** 2 + (y / b) ** 2) - 1) * np.minimum(a, b)


# ---------------------------------------------------------------- leaves

_t = np.linspace(0, 1, 2001)
_LEAF_MAX = (_t ** 0.6 * (1 - _t) ** 0.3).max()


def leaf_sdf(fr, sc=1.0, T=0.9):
    """Small rounded ʻōhiʻa leaf, widest beyond the middle with a blunt tip
    (obovate), cupped upward, a raised midrib and a short petiole running
    into the twig."""
    L, W = LEAF_L * sc, LEAF_W * sc

    def f(x, y, z):
        u, v, w = local(fr, x, y, z)
        t = u / L
        tc = np.clip(t, 0, 1)
        half = W * tc ** 0.6 * (1 - tc) ** 0.3 / _LEAF_MAX + 1e-3
        o = np.maximum(np.abs(v) - half, np.maximum(-u, u - L))
        h = 0.10 * v ** 2 / sc + 0.8 * sc * tc ** 2         # cupped, tip lifts
        d = soft_band(o, np.abs(w - h) - T / 2, 2.0, T / 2)
        rib = np.maximum(np.abs(v) - 0.36, o + 0.8)
        d = smin(d, soft_band(rib, np.abs(w - h - 0.4) - 0.3, 0.36, 0.3), 0.25)
        stalk = capsule_chain(u, v, w, np.array([[-2.4, 0, 0.0], [0.4, 0, 0.0]]),
                              np.array([0.6, 0.5]))
        return smin(d, stalk, 0.4)
    return f, box(fr, (-3.2, -W - 1, -2.5), (L + 1, W + 1, 3.5))


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

def lehua(c, axis, seed=0, avoid=(), n_fl=11, per=12):
    """One lehua blossom: a cluster of flowers on a small knob. Each flower
    is a stalk swelling into a rounded, fuzzy cup, with a fan of long
    stamens tipped with bead-like anthers; a few closed buds sit among
    them. Returned as tubes (points, radii).

    Printability: flowers point at least 40 degrees above horizontal, so
    the cups (which widen at about 30 degrees) rise from their stalks;
    stamens leave at least STAMEN_EL and curl upward, and each anther
    swells gradually, so every tip is the highest point of its stamen.
    Stamens passing through `avoid` spheres (the bird) are left out."""
    rng = np.random.default_rng(seed)
    _, e3, e1, e2 = frame(c, axis, np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0]))
    tubes = [(np.array([c - 1.5 * axis, c + 0.6 * axis]), np.array([1.6, 2.2]))]   # knob
    el_min = np.radians(40)
    for i in range(n_fl + 3):
        h = 1 - (i + 0.5) / (n_fl + 3) * 1.15
        a = i * GOLDEN_ANGLE + seed
        n = h * e3 + np.sqrt(max(1 - h * h, 0)) * (np.cos(a) * e1 + np.sin(a) * e2)
        if np.arcsin(np.clip(n[2], -1, 1)) < el_min:
            hz = unit(n * [1, 1, 0] + 1e-6)
            n = np.cos(el_min) * hz + np.sin(el_min) * UP
        base = c + 1.4 * n
        if i % 5 == 2:                                     # a closed bud
            pts = base + np.linspace(0, 3.6, 7)[:, None] * n
            tubes.append((pts, np.array([0.5, 0.55, 0.75, 1.1, 1.3, 1.25, 0.8])))
            continue
        L0 = 2.6 + 0.6 * rng.random()                      # stalk, then the cup
        pts = base + np.linspace(0, L0 + 2.2, 8)[:, None] * n
        tubes.append((pts, np.array([0.5, 0.52, 0.56, 0.75, 1.05, 1.35, 1.45, 1.25])))
        top = base + (L0 + 1.9) * n
        u1 = unit(np.cross(n, e3 + 1e-3 * e1))
        u2 = np.cross(n, u1)
        for k in range(per):
            g = k * GOLDEN_ANGLE + a
            sp = 0.95 * np.sqrt((k + 0.5) / per)
            q = unit(n + sp * (np.cos(g) * u1 + np.sin(g) * u2))
            if np.arcsin(np.clip(q[2], -1, 1)) < STAMEN_EL:
                hz = unit(q * [1, 1, 0] + 1e-6)
                q = np.cos(STAMEN_EL) * hz + np.sin(STAMEN_EL) * UP
            Ls = 7.5 + 2.2 * rng.random()
            p0 = top + 0.5 * q
            curl = 0.14 * Ls * np.clip(1 - q[2], 0, 1) * UP
            st = bezier(p0, p0 + 0.35 * Ls * q, p0 + 0.7 * Ls * q + 0.4 * curl, p0 + Ls * q + curl, n=12)
            if any(np.min(np.linalg.norm(st - ac, axis=1)) < ar for ac, ar in avoid):
                continue
            s_ = np.linspace(0, 1, 12)
            r0, r1, ra = STAMEN_R
            rad = np.where(s_ < 0.8, r0 + (r1 - r0) * s_ / 0.8,
                           r1 + (ra - r1) * np.clip((s_ - 0.8) / 0.12, 0, 1))
            rad[-1] = 0.8 * ra
            tubes.append((st, rad))
    return tubes


def bud_cluster(c, axis, n=12, seed=1):
    """A tight cluster of round, fuzzy flower buds, like popcorn, on short
    stalks. Each bud is a teardrop that widens along its stalk."""
    rng = np.random.default_rng(seed)
    tubes = [(np.array([c - 1.0 * axis, c + 0.8 * axis]), np.array([1.0, 1.3]))]
    _, e3, e1, e2 = frame(c, axis, np.array([1.0, 0, 0]))
    for i in range(n):
        h = 1 - (i + 0.5) / n * 0.95
        a = i * GOLDEN_ANGLE + seed
        d = unit(h * e3 + np.sqrt(1 - h * h) * (np.cos(a) * e1 + np.sin(a) * e2) + 0.6 * UP)
        L = 2.4 + 0.9 * rng.random()
        pts = np.array([c + (L * s) * d for s in np.linspace(0, 1, 7)])
        tubes.append((pts, np.array([0.5, 0.55, 0.7, 0.95, 1.15, 1.2, 0.85])))
    return tubes


# ------------------------------------------------------------------ bird

def iiwi(P, f, target, tail_tip, S=1.0):
    """An ʻiʻiwi (scarlet honeycreeper) perched at P on a branch, facing f,
    reaching its long down-curved bill into a blossom at `target`. Plump
    body, round head with a bead eye, folded wings of layered feather
    tracts, a notched tail resting on the branch behind, legs and toes
    gripping the branch. S scales the whole bird (1.0: about 25 mm long).

    Printability: the belly sits just above the branch on the legs, the
    tail tip rests on the branch and the bill tip is in the blossom, so
    every low point of the bird is held up by something."""
    f = unit(f * [1, 1, 0])
    s = np.cross(UP, f)
    M = np.vstack([f, s, UP])                               # world -> local rows

    def L(p):
        return P + S * (p[0] * f + p[1] * s + p[2] * UP)
    hc = L([5.4, 0, 9.6])
    pitch = np.radians(20)
    bill_end = unit(f - 0.9 * UP)
    p0 = hc + 2.6 * S * unit(target - hc + 2.0 * UP)
    Lb = np.linalg.norm(target - p0)
    bill = bezier(p0, p0 + 0.4 * Lb * unit(target - p0 + 0.35 * Lb * UP),
                  target - 0.3 * Lb * bill_end, target, n=24)
    bill_r = np.linspace(1.05 * S, 0.32, 24)
    rump = L([-5.5, 0, 4.2])
    tu = unit(tail_tip - rump)
    tL = np.linalg.norm(tail_tip - rump)
    tw = unit(UP - (UP @ tu) * tu)
    tfr = (rump, tu, np.cross(tw, tu), tw)
    eyes = [hc + 3.0 * S * unit(0.45 * f + sg * 0.8 * s + 0.35 * UP) for sg in (1, -1)]

    def ell(a, b, h, c_, ax, pt):
        ca, sa = np.cos(pt), np.sin(pt)
        a2, h2 = (a - c_[0]) * ca + (h - c_[2]) * sa, -(a - c_[0]) * sa + (h - c_[2]) * ca
        r = np.sqrt((a2 / ax[0]) ** 2 + ((b - c_[1]) / ax[1]) ** 2 + (h2 / ax[2]) ** 2)
        return (r - 1) * min(ax)

    def fn(x, y, z):
        q = np.stack([x - P[0], y - P[1], z - P[2]], -1) @ M.T
        a, b, h = q[..., 0] / S, q[..., 1] / S, q[..., 2] / S
        d = ell(a, b, h, (0.0, 0, 5.0), (6.6, 4.0, 4.3), pitch)               # body
        d = smin(d, np.sqrt((a - 5.4) ** 2 + b ** 2 + (h - 9.6) ** 2) - 3.3, 2.2)   # head
        for sg in (1, -1):                                  # folded wings, lying along the back
            wing = 1e9
            for off, ln, ht, th, dz in ((0.0, 4.4, 2.9, 0.9, 0.0), (-1.8, 5.0, 2.5, 0.8, -0.3),
                                         (-3.6, 5.4, 1.9, 0.7, -0.7)):
                wing = np.minimum(wing, ell(a, b, h, (-1.8 + off, sg * (3.05 + 0.08 * off), 6.6 + dz + 0.25 * off),
                                            (ln, th, ht), pitch))
            d = smin(d, wing, 0.7)
        d = d * S
        tu_, tv_, tw_ = local(tfr, x, y, z)                 # tail
        t = np.clip(tu_ / tL, 0, 1)
        half = S * (1.5 + 0.6 * t)
        notch = tL - 0.9 * S * (1 - np.clip(np.abs(tv_) / half, 0, 1))
        o = np.maximum(np.abs(tv_) - half, np.maximum(-tu_, tu_ - notch))
        d = smin(d, soft_band(o, np.abs(tw_) - 0.5 * S, 1.0, 0.5 * S), 0.8 * S)
        d = smin(d, capsule_chain(x, y, z, bill, bill_r), 0.6 * S)
        for e in eyes:
            d = smin(d, np.sqrt((x - e[0]) ** 2 + (y - e[1]) ** 2 + (z - e[2]) ** 2) - 0.6 * S, 0.15)
        for sg in (1, -1):                                  # legs and toes
            leg = np.array([L([-0.6, sg * 1.4, 2.4]), L([-0.4, sg * 1.5, 0.6]), L([-0.3, sg * 1.5, -0.6])])
            d = smin(d, capsule_chain(x, y, z, leg, S * np.array([0.6, 0.55, 0.55])), 0.4)
            for tt in (L([2.0, sg * 1.7, -0.4]), L([-2.0, sg * 1.6, -0.4]), L([-0.3, sg * 2.6, -1.2])):
                d = smin(d, capsule_chain(x, y, z, np.array([leg[1], tt]), S * np.array([0.5, 0.4])), 0.3)
        return d
    pts = np.vstack([bill, [L([a_, b_, h_]) for a_ in (-13, 10) for b_ in (-6, 6) for h_ in (-2, 14)], [tail_tip]])
    keep = [(L([0, 0, 5.0]), 7.6 * S), (hc, 6.5 * S), (rump + 0.5 * tL * tu, 3.2 * S)] + \
        [(p, 1.8 * S) for p in bill[::3]]
    return fn, (pts.min(0) - 3.0 * S, pts.max(0) + 3.0 * S), keep


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

    # the blossoms: each on a twig rising out of a branch end, standing
    # clear of a cushion of leafy sprigs
    poms = []
    for j, (br, out) in enumerate(ends):
        axis = unit(0.45 * out + UP)
        c = br[-1] + 8.5 * axis + 0.5 * out
        poms.append((c, axis, j))
    bird = place_bird(ends[BIRD_BRANCH][0], np.linspace(BRANCH_R[BIRD_BRANCH], 1.25, 50),
                      poms[BIRD_BRANCH][0], BIRD_S)
    keep = bird[2]

    def hits(pts, pad=1.6):
        return any(np.min(np.linalg.norm(pts - kc, axis=1)) < kr + pad for kc, kr in keep)

    fine = []
    for (br, out), (c, axis, j) in zip(ends, poms):
        tip = br[-1]
        side = unit(np.cross(UP, out))
        tw = bezier(tip, tip + 2.5 * unit(br[-1] - br[-3]), c - 3.5 * axis, c - 1.0 * axis, n=16)
        tubes.append((tw, np.linspace(1.25, 1.1, 16)))
        fine += lehua(c, axis, seed=j, avoid=keep if j == BIRD_BRANCH else ())
        a_out = np.arctan2(out[1], out[0])
        roots = (br[-1], br[-6]) if j < 4 else (line[-1], line[-6])
        fan = (-2.0, -1.05, -0.2, 0.7, 1.5, 2.6) if j < 4 else (0.3, 1.35, 2.4, 3.45, 4.5, 5.55)
        sprigs = []
        for m, da in enumerate(fan):
            a = a_out + da + 0.15 * np.sin(3 * j + m)
            sd = unit(np.cos(np.radians(24)) * np.array([np.cos(a), np.sin(a), 0])
                      + np.sin(np.radians(24)) * UP)
            sprigs.append((roots[m % 2], sd, 10.5 + 1.5 * np.sin(2 * j + m), (j + m) % 4 == 1))
        if j < 4:                                          # sprigs further back along the branch
            for m, (f_, sgn) in enumerate(((0.5, 1), (0.72, -1))):
                sd = unit(0.8 * sgn * side + 0.4 * out + 0.55 * UP)
                sprigs.append((br[int(f_ * (len(br) - 1))], sd, 11.0 - 1.5 * m, (j + m) % 2 == 0))
        for p, sd, L, buds in sprigs:
            sp = bezier(p, p + 2.5 * unit(sd + 0.6 * UP), p + L * sd - 3.0 * sd, p + L * sd, n=18)
            if hits(sp):
                continue
            tubes.append((sp, np.linspace(1.05, 0.8, 18)))
            sprig_leaves(things, sp, sd)
            if buds and not hits(sp[-1:] + 0.5 * sd, 3.5):
                tubes += bud_cluster(sp[-1] + 0.5 * sd, unit(sd + UP), seed=int(10 * L) + j)
            else:
                things.append(("liko", (sp[-1] - 0.3 * sd, unit(0.35 * sd + UP))))
    # leaf pairs along the branches themselves
    for j, (br, out) in enumerate(ends[:4]):
        sprig_leaves(things, br[int(0.4 * len(br)):], unit(br[-1] - br[int(0.4 * len(br))]),
                     nodes=((0.3, 1.0), (0.6, 1.0)), rosette=False)

    kept = [t for t in things if not clips_pom(t, poms) and not hits(thing_pts(t), 1.2)]
    return tubes, kept, poms, fine, bird


def place_bird(br, rad, c, S=1.0):
    """Perch the ʻiʻiwi where its bill reaches the branch's blossom best:
    about 7.5 mm of bill (times S), head no lower than 2 mm under the
    flowers it sips from."""
    best = None
    for i in range(12, 48):
        P = br[i] + rad[i] * UP
        f = unit((c - P) * [1, 1, 0])
        target = c + 5.0 * unit(-f * np.cos(np.radians(40)) + UP * np.sin(np.radians(40)))   # a low flower's cup
        head = P + S * (5.4 * f + 9.6 * UP)
        Lb = np.linalg.norm(target - head) - 2.6 * S
        score = abs(Lb - 7.5 * S) + 2.0 * max(target[2] - head[2] - 2.0, 0)
        if best is None or score < best[0]:
            best = (score, i, P, f, target)
    _, i, P, f, target = best
    k = max(i - int(14 * S), 0)                                   # tail rests on the branch behind
    tail_tip = br[k] + (rad[k] + 0.1) * UP
    return iiwi(P, f, target, tail_tip, S)


def sprig_leaves(things, sp, sd, nodes=((0.3, 1.0), (0.55, 1.0), (0.78, 0.85)), rosette=True):
    """Opposite leaf pairs along a sprig, each pair turned 90 degrees from
    the last, leaves angled forward and up; at the tip, a rosette of
    smaller leaves crowding round the bud."""
    side = unit(np.cross(UP, sd * [1, 1, 0] + 1e-6))
    w = unit(np.cross(sd, side))
    if rosette:
        nodes = nodes + ((0.93, 0.7),)
    for k, (f_, sc) in enumerate(nodes):
        q = sp[int(f_ * (len(sp) - 1))]
        ax = side if k % 2 == 0 else w
        lift = 0.45 + (0.35 if f_ > 0.9 else 0)
        for sgn in (1, -1):
            ld = unit(0.75 * sgn * ax + 0.55 * sd + lift * UP)
            if ld[2] < 0.42:                              # keep them rising
                ld = unit(ld * [1, 1, 0] * 0.9 / max(np.linalg.norm(ld[:2]), 1e-6) + 0.45 * UP)
            things.append(("leaf", frame(q + 1.3 * ld, ld), sc))


def thing_pts(thing):
    if thing[0] == "leaf":
        base, u, _, _ = thing[1]
        return base + np.linspace(0, 1.0, 8)[:, None] * LEAF_L * thing[2] * u
    base, u = thing[1]
    return base + np.linspace(0, 1.0, 5)[:, None] * 4.6 * u


def clips_pom(thing, poms):
    """True for a leaf or liko that would poke into a blossom's stamens."""
    pts = thing_pts(thing)
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
    tubes, things, poms, fine, bird = tree_parts()
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
    for kind, *spec in things:
        f, (blo, bhi) = leaf_sdf(*spec) if kind == "leaf" else liko_sdf(*spec[0])
        add(vol, xs, ys, zs, blo, bhi, f, k=0.4, chunk=8)
    say(f"  {len(things)} leaves and liko {time.time() - t0:.0f}s")
    for pts, rr in fine:
        add_tube(vol, xs, ys, zs, pts, rr, k=0.3)
    say(f"  {len(poms)} blossoms ({len(fine)} stamens, cups and buds) {time.time() - t0:.0f}s")
    f, (blo, bhi), _ = bird
    add(vol, xs, ys, zs, blo, bhi, f, k=0.5, chunk=8)
    say(f"  ʻiʻiwi {time.time() - t0:.0f}s")
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
