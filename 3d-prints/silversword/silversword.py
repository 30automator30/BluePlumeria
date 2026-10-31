#!/usr/bin/env python3
"""ʻĀhinahina, the Haleakalā silversword (resin / MSLA, Saturn 4 Ultra),
about 150 mm tall. One part, printed straight on the plate:

  plinth   an oval of red cinder ground: a low mound with a pebbled
           surface and scattered cinders, its sides flaring to the plate.
  rosette  a ball of about 200 silver needle leaves. Upper needles rise and
           curve upward; the lower ones droop out until their tips rest on
           the cinders.
  spike    the bloom spike, a tall egg-shaped cob covered in about 150
           daisy-like flower heads in a golden spiral. Each head is a cup on
           a short stalk, a domed centre pitted with florets and a ring of
           tubular ray petals; the heads are smaller toward both ends, with
           buds at the tip and slim green bracts at the foot.
  rosettes three small young rosettes around the plinth.
  nēnē     a Hawaiian goose resting on one leg, looking up at the bloom:
           furrowed neck, layered and barred wing feathers, short tail,
           webbed foot.

Printability: flower heads tilt at least HEAD_EL above horizontal, their
cups widen at no more than 35 degrees, and every ray petal is cupped
forward just enough to point at least RAY_EL upward; floret texture is
cut in as pits, never raised. Needles rise, or droop to the ground. The
goose stands on the leg under its lowest point.

Usage
  python3 silversword.py --voxel 0.25 --preview         # quick look
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("fractal-plumeria", "plumeria-ring", "hibiscus-bonsai", "ohia-bonsai"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import lattice_plumeria as lp  # noqa: E402
from bonsai_hibiscus import box, frame, local, unit  # noqa: E402
from fractal_plumeria import capsule_chain, mesh_volume  # noqa: E402
from lattice_plumeria import BASE_CUT, add, bezier, soft_band  # noqa: E402
from ohia_bonsai import add_tube, oval  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))
UP = np.array([0, 0, 1.0])

# ---------------------------------------------------------------- layout
PLINTH = (62.0, 42.0)                # half-axes of the oval
PL_H = 5.0                           # plinth height at the rim
CENTRE = np.array([-10.0, 4.0])      # the silversword
CROWN_H = 17.0                       # rosette crown above the ground
SPIKE = (30.0, 150.0)                # bloom spike: bottom and tip heights
HEAD_GAP = 6.8                       # spacing of flower heads on the spike
HEAD_EL = np.radians(40)             # heads face at least this far up
RAY_EL = np.radians(8)               # ray petals point at least this far up
SMALL = [(np.array([-46.0, 22.0]), 0.42), (np.array([34.0, 26.0]), 0.36),
         (np.array([-40.0, -26.0]), 0.3)]          # small rosettes: (where, scale)
NENE = (np.array([36.0, -20.0]), 1.0)               # goose: (foot, scale)


def ground(x, y):
    """Height of the cinder ground: a low mound under the silversword."""
    r = np.sqrt(((x - CENTRE[0]) / 40.0) ** 2 + ((y - CENTRE[1]) / 30.0) ** 2)
    return PL_H + 2.2 * np.exp(-r ** 2 * 1.6) + 0.8 * np.exp(-(r / 0.6) ** 2)


def plinth_sdf(x, y, z):
    rim = oval(x, y, PLINTH[0] + 1.6 * np.clip((PL_H - z) / PL_H, 0, 1), PLINTH[1] + 1.6 * np.clip((PL_H - z) / PL_H, 0, 1))
    bumps = 0.0
    for k in range(9):                                   # irregular cinder texture
        a_ = 2.39996 * k + 0.3
        fq = 1.1 + 0.55 * k
        bumps = bumps + 0.32 / (1 + 0.45 * k) * np.sin(fq * (np.cos(a_) * x + np.sin(a_) * y) + 1.7 * k) \
            * np.sin(0.8 * fq * (np.cos(a_ + 1.2) * x + np.sin(a_ + 1.2) * y) + 0.9 * k)
    top = z - ground(x, y) - bumps
    d = np.maximum(rim, np.maximum(top, -z))
    rng = np.random.default_rng(3)                       # scattered cinders, half sunk
    for _ in range(150):
        a, rr = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0.15, 0.93))
        px, py = PLINTH[0] * rr * np.cos(a), PLINTH[1] * rr * np.sin(a)
        if np.hypot(px - CENTRE[0], py - CENTRE[1]) < 34:
            continue
        r0 = rng.uniform(0.6, 1.4)
        pz = ground(px, py)
        d = np.minimum(d, np.sqrt((x - px) ** 2 + (y - py) ** 2 + ((z - pz) / 0.8) ** 2) * 0.8 - r0 * 0.8)
    return d


# --------------------------------------------------------------- rosette

def needle(p0, d0, L, curl, r0=0.75, n=16):
    """One silver leaf: a narrow, thick needle, widest a third of the way,
    tapering to a sharp tip, curving upward by `curl` radians."""
    d0 = unit(d0)
    side = unit(np.cross(d0, UP)) if abs(d0[2]) < 0.99 else np.array([1.0, 0, 0])
    up_ = unit(np.cross(side, d0))
    d1 = unit(np.cos(curl) * d0 + np.sin(curl) * up_)
    pts = bezier(p0, p0 + 0.4 * L * d0, p0 + L * (0.4 * d0 + 0.4 * d1), p0 + L * (0.4 * d0 + 0.6 * d1), n=n)
    s = np.linspace(0, 1, n)
    rad = r0 * (0.75 + 0.35 * np.sin(np.pi * np.clip(s / 0.66, 0, 1)) ** 0.8) * (1 - s) ** 0.75 + 0.12
    return pts, rad


def rosette(c, scale=1.0, n_up=230, n_skirt=80, seed=0):
    """A ball of silver needles round a crown at c: rising needles spread
    evenly over the upper sphere and curve gently up; a skirt of drooping
    ones bows out round the lower half and rests its tips on the ground."""
    rng = np.random.default_rng(seed)
    tubes = []
    gz = ground(c[0], c[1])
    crown = np.array([c[0], c[1], gz + CROWN_H * scale])
    r0 = 0.75 * max(scale, 0.55)

    def profile(n):
        s_ = np.linspace(0, 1, n)
        return r0 * (0.75 + 0.35 * np.sin(np.pi * np.clip(s_ / 0.66, 0, 1)) ** 0.8) * (1 - s_) ** 0.75 + 0.13
    for i in range(n_up):
        # evenly over the upper sphere (equal area), from 6 to 88 degrees up
        t = (i + 0.5) / n_up
        el = np.arcsin(np.sin(np.radians(6)) + t * (np.sin(np.radians(88)) - np.sin(np.radians(6))))
        a = i * GOLDEN_ANGLE + seed
        d = np.cos(el) * np.array([np.cos(a), np.sin(a), 0]) + np.sin(el) * UP
        L = scale * (29 + 5 * rng.random())
        p0 = crown + scale * 2.2 * d
        pts, _ = needle(p0, d, L, curl=np.radians(3 + 7 * t))
        tubes.append((pts, profile(len(pts))))
    for i in range(n_skirt):
        a = i * GOLDEN_ANGLE + 0.4 + seed
        out = np.array([np.cos(a), np.sin(a), 0])
        dn = np.radians(4 + 26 * ((i * 0.618) % 1.0))      # leaves the crown 4-30 degrees down
        d0 = np.cos(dn) * out - np.sin(dn) * UP
        reach = scale * (29 + 5 * rng.random())
        p0 = crown + scale * 2.2 * d0
        tip = np.array([c[0], c[1], 0]) + reach * out
        tip[2] = ground(tip[0], tip[1]) - 0.15             # rests on the cinders
        k = np.linalg.norm(tip - p0)
        pts = bezier(p0, p0 + 0.6 * k * d0, tip + 0.35 * k * unit(0.25 * out + UP) - 0.1 * k * out, tip, n=16)   # bows out, then down
        tubes.append((pts, profile(16) + 0.02))
    core = (np.array([crown - [0, 0, CROWN_H * scale + 0.5], crown + [0, 0, scale * 2.0]]),
            np.array([scale * 4.5, scale * 3.8]))
    tubes.append(core)
    return tubes, crown


# ---------------------------------------------------------------- spike

def spike_r(z):
    """Radius of the spike's core: a tall egg, narrow at the foot, widest a
    little below the middle, tapering to the tip."""
    t = np.clip((z - SPIKE[0]) / (SPIKE[1] - SPIKE[0]), 0, 1)
    return 2.4 + 10.4 * np.sin(np.pi * t ** 0.8) ** 0.9 * (1 - 0.35 * t)


def spike_heads():
    """Flower-head positions on the spike: an even golden spiral over its
    surface. Returns (base point, facing, size) for each head."""
    z = np.linspace(SPIKE[0] + 2.0, SPIKE[1] - 4.0, 2000)
    r = spike_r(z)
    dr = np.gradient(r, z)
    dA = 2 * np.pi * r * np.sqrt(1 + dr ** 2)
    A = np.concatenate([[0], np.cumsum(0.5 * (dA[1:] + dA[:-1]) * np.diff(z))])
    n = int(A[-1] / HEAD_GAP ** 2)
    heads = []
    for k in range(n):
        zk = np.interp((k + 0.5) / n * A[-1], A, z)
        a = k * GOLDEN_ANGLE
        rk, gk = spike_r(zk), np.interp(zk, z, dr)
        out = np.array([np.cos(a), np.sin(a), 0])
        nrm = unit(out - gk * UP)                         # surface normal
        el = np.arcsin(np.clip(nrm[2], -1, 1))
        face = unit(np.cos(max(el, HEAD_EL)) * out + np.sin(max(el, HEAD_EL)) * UP)
        t = (zk - SPIKE[0]) / (SPIKE[1] - SPIKE[0])
        size = 0.62 + 0.38 * np.sin(np.pi * np.clip(t * 1.05, 0, 1)) ** 0.6
        base = np.array([CENTRE[0], CENTRE[1], zk]) + (rk - 0.6) * out
        heads.append((base, face, size, out))
    return heads


def head_parts(base, n, size, out, seed=0):
    """One flower head, facing n. Returns (tubes, dome, pits): the stalk and
    cup as one tube, the ray petals as tubes, the dome as (centre, radius,
    depth) and the pit centres for the floret texture."""
    rng = np.random.default_rng(seed)
    rd = 2.05 * size
    up_p = unit(UP - (UP @ n) * n)                        # up, in the head's plane
    side = np.cross(n, up_p)
    q = base + (1.0 + 2.4 * size) * n                     # dome centre
    # stalk out of the core, then a cup widening at 34 degrees to the dome
    cup_len = (rd - 0.6) / np.tan(np.radians(34))
    pts = np.array([base - 1.4 * out, base + 0.4 * n, q - cup_len * n, q - 0.5 * cup_len * n, q])
    tubes = [(pts, np.array([0.8, 0.65, 0.6, 0.6 + 0.5 * (rd - 0.6), rd]))]
    n_ray = 14
    for i in range(n_ray):
        a = 2 * np.pi * (i + 0.3 * rng.random()) / n_ray
        r_dir = np.cos(a) * up_p + np.sin(a) * side
        root = q + 0.9 * rd * r_dir - 0.2 * n
        for cup in np.radians(np.arange(22, 82, 4)):     # flattest cup that still rises
            d = np.cos(cup) * r_dir + np.sin(cup) * n
            if np.arcsin(np.clip(d[2], -1, 1)) >= RAY_EL:
                break
        L = size * (2.9 + 0.4 * rng.random())
        d2 = unit(d + 0.35 * UP)                          # tips flick up a little
        ray = bezier(root, root + 0.4 * L * d, root + 0.75 * L * d, root + 0.6 * L * d + 0.4 * L * d2, n=8)
        tubes.append((ray, size * np.array([0.5, 0.52, 0.5, 0.46, 0.42, 0.36, 0.28, 0.2])))
    pits = []
    for k in range(19):
        rr = 0.72 * rd * np.sqrt((k + 0.5) / 19)
        a = k * GOLDEN_ANGLE
        p = q + rr * (np.cos(a) * up_p + np.sin(a) * side)
        h = 0.9 * size * np.sqrt(max(1 - (rr / rd) ** 2, 0))
        pits.append(p + (h + 0.12 * size) * n)
    return tubes, (q, rd, 0.9 * size, n), pits


def spike_tubes():
    tubes, domes, pits = [], [], []
    z = np.linspace(SPIKE[0] - 9.0, SPIKE[1], 120)
    core = np.column_stack([np.full_like(z, CENTRE[0]), np.full_like(z, CENTRE[1]), z])
    tubes.append((core, np.maximum(spike_r(z), 2.4)))
    # tip: a few closed buds round the point
    top = np.array([CENTRE[0], CENTRE[1], SPIKE[1]])
    tubes.append((np.array([top - [0, 0, 2.0], top + [0, 0, 1.6]]), np.array([2.2, 1.2])))
    for k in range(6):
        a = k * GOLDEN_ANGLE
        d = unit(0.8 * np.array([np.cos(a), np.sin(a), 0]) + UP)
        p = top - [0, 0, 3.0] + 1.5 * d
        tubes.append((np.array([p, p + 1.6 * d, p + 2.6 * d]), np.array([0.6, 1.1, 0.7])))
    for k, (base, n, size, out) in enumerate(spike_heads()):
        t_, dm, pt = head_parts(base, n, size, out, seed=k)
        tubes += t_
        domes.append(dm)
        pits += pt
    # slim green bracts round the foot of the spike, rising
    for k in range(26):
        a = k * GOLDEN_ANGLE
        out = np.array([np.cos(a), np.sin(a), 0])
        zk = SPIKE[0] - 6.0 + 9.0 * (k / 26)
        p0 = np.array([CENTRE[0], CENTRE[1], zk]) + (spike_r(zk) - 0.5) * out
        el = np.radians(48 + 18 * (k % 3))
        tubes.append(needle(p0, np.cos(el) * out + np.sin(el) * UP, 9.0 + 3.0 * (k % 2), np.radians(20), r0=0.55, n=12))
    return tubes, domes, pits


def dome_sdf(q, rd, depth, n):
    def f(x, y, z):
        p = np.stack([x - q[0], y - q[1], z - q[2]], -1)
        h = p @ n
        rr = np.sqrt(np.maximum((p * p).sum(-1) - h * h, 0))
        return np.sqrt((rr / rd) ** 2 + (h / depth) ** 2) * min(rd, depth) - min(rd, depth)
    return f, (q - rd - 0.5, q + rd + 0.5)


# ------------------------------------------------------------------ nēnē

def nene(P, f, S=1.0):
    """A nēnē (Hawaiian goose), about 45 mm tall, resting on one leg with
    the other tucked up: furrowed neck, black head and short bill with bead
    eyes, folded wings in layered rows with barred edges, a short tail, and
    a webbed foot flat on the ground.

    Printability: the standing leg is under the body's lowest point, the
    foot lies on the ground, the bill is level and the tail rises."""
    f = unit(f * [1, 1, 0])
    s = np.cross(UP, f)
    M = np.vstack([f, s, UP])
    pitch = np.radians(8)                                 # chest a little up

    def L(p):
        return P + S * (p[0] * f + p[1] * s + p[2] * UP)
    B, AX = np.array([0.0, 0, 17.0]), np.array([13.0, 6.6, 7.4])
    # the body's lowest point (the standing leg goes there)
    th = np.linspace(0, 2 * np.pi, 721)
    ca, sa = np.cos(pitch), np.sin(pitch)
    ea, eh = AX[0] * np.cos(th), AX[2] * np.sin(th)
    low = np.argmin(ea * sa + eh * ca)
    low_a, low_h = B[0] + ea[low] * ca - eh[low] * sa, B[2] + ea[low] * sa + eh[low] * ca
    neck = bezier(np.array([9.5, 0, 20.0]), np.array([13.0, 0, 26.0]), np.array([11.0, 0, 33.0]),
                  np.array([13.5, 0, 38.0]), n=30)
    neck_w = np.array([L(p) for p in neck])
    neck_r = S * np.linspace(3.1, 2.1, 30)
    hc = np.array([15.0, 0, 39.8])
    bill = np.array([L([17.8, 0, 39.3]), L([19.4, 0, 39.3]), L([20.8, 0, 39.45])])
    eyes = [L([16.2, sg * 1.9, 40.6]) for sg in (1, -1)]
    leg = np.array([L([low_a, 0, low_h + 2.0]), L([low_a - 0.4, 0, 6.0]), L([low_a, 0, 0.9])])
    foot = L([low_a, 0, 0.45])

    def ell(a, b, h, c_, ax, pt):
        c2, s2 = np.cos(pt), np.sin(pt)
        a2, h2 = (a - c_[0]) * c2 + (h - c_[2]) * s2, -(a - c_[0]) * s2 + (h - c_[2]) * c2
        r = np.sqrt((a2 / ax[0]) ** 2 + ((b - c_[1]) / ax[1]) ** 2 + (h2 / ax[2]) ** 2)
        return (r - 1) * min(ax)

    def fn(x, y, z):
        q = np.stack([x - P[0], y - P[1], z - P[2]], -1) @ M.T
        a, b, h = q[..., 0] / S, q[..., 1] / S, q[..., 2] / S
        d = ell(a, b, h, B, AX, pitch)
        for sg in (1, -1):                                # folded wings: four tiers
            for k, (off, ln, ht, dz) in enumerate(((3.0, 6.0, 4.2, 2.6), (-0.5, 7.0, 3.8, 2.8),
                                                   (-4.0, 7.2, 3.3, 3.1), (-7.5, 6.4, 2.6, 3.4))):
                w = ell(a, b, h, (off, sg * (4.9 - 0.25 * k), B[2] + dz), (ln, 1.3, ht), pitch - 0.05)
                d = smin(d, w, 0.9) if k == 0 else smin(d, w, 0.25)
        # barred feather edges: shallow scalloped grooves across back and flanks
        row = np.abs(((a + 0.3 * np.abs(b) ** 1.3) / 2.6) % 1.0 - 0.5) * 2.6
        back = (h > B[2] - 1.5) & (a < 8)
        d = np.where(back, np.maximum(d, -np.maximum(row - 0.12, -d - 0.22)), d)
        tail = ell(a, b, h, (-14.0, 0, 19.6), (4.5, 2.6, 1.3), np.radians(-14))
        d = smin(d, tail, 0.9)
        d = d * S
        # neck with furrows: the feathers part into dark grooves along it
        dn = capsule_chain(x, y, z, neck_w, neck_r)
        tq = np.stack([x, y, z], -1)
        k_ = np.argmin(np.stack([np.sum((tq - p) ** 2, -1) for p in neck_w[::3]]), 0) * 3
        cen = neck_w[k_]
        rel = tq - cen
        ang = np.arctan2(rel @ s, rel @ f)
        groove = 0.2 * S * np.abs(np.sin(9 * ang)) ** 3
        dn = dn + np.where(dn > -0.8 * S, groove, 0)
        d = smin(d, dn, 1.6 * S)
        hd = np.sqrt(((a - hc[0]) / 3.4) ** 2 + (b / 2.4) ** 2 + ((h - hc[2]) / 2.7) ** 2) * 2.4 - 2.4
        d = smin(d, hd * S, 1.2 * S)
        d = smin(d, capsule_chain(x, y, z, bill, S * np.array([1.15, 0.85, 0.5])), 0.5 * S)
        for e in eyes:
            d = smin(d, np.sqrt((x - e[0]) ** 2 + (y - e[1]) ** 2 + (z - e[2]) ** 2) - 0.5 * S, 0.15)
        d = smin(d, capsule_chain(x, y, z, leg, S * np.array([1.6, 1.1, 1.0])), 0.6 * S)
        for ang_ in (-0.45, 0.0, 0.45):                   # three toes, webbed
            dirn = np.cos(ang_) * f + np.sin(ang_) * s
            toe = np.array([foot, foot + S * 5.0 * dirn])
            d = smin(d, capsule_chain(x, y, z, toe, S * np.array([0.6, 0.4])), 0.3)
        fq = np.stack([x - foot[0], y - foot[1], z - foot[2]], -1)
        fa, fb, fh = fq @ f, fq @ s, fq @ UP
        web = np.maximum(np.maximum(np.abs(fb) - 0.48 * np.clip(fa, 0, None) - 0.4, np.maximum(-fa, fa - 4.6 * S)),
                         np.abs(fh) - 0.3 * S)
        d = np.minimum(d, web)
        return d
    corners = np.array([L([a_, b_, h_]) for a_ in (-20, 24) for b_ in (-10, 10) for h_ in (-1, 45)])
    return fn, (corners.min(0) - 2.0, corners.max(0) + 2.0), (L(B), 15.0 * S)


# ------------------------------------------------------------------ build

def parts():
    tubes, domes, pits = spike_tubes()
    t_, crown = rosette(CENTRE, 1.0, seed=0)
    tubes += t_
    for k, (c, sc) in enumerate(SMALL):
        tubes += rosette(c, sc, n_up=int(110 * sc + 30), n_skirt=int(40 * sc + 12), seed=k + 1)[0]
    P = np.array([NENE[0][0], NENE[0][1], ground(*NENE[0]) - 0.3])
    goose = nene(P, np.append(CENTRE - NENE[0], 0), NENE[1])
    # keep needles out of the goose
    gc_, gr = goose[2]
    tubes = [t for t in tubes if np.min(np.linalg.norm(t[0] - gc_, axis=1)) > gr + 2.0]
    return tubes, domes, pits, goose


def bounds():
    return np.array([-PLINTH[0] - 3, -PLINTH[1] - 3, -0.6]), np.array([PLINTH[0] + 3, PLINTH[1] + 3, SPIKE[1] + 4])


def build(voxel, stems=(), quiet=False, region=None):
    say = (lambda *a: None) if quiet else print
    t0 = time.time()
    tubes, domes, pits, goose = parts()
    lo, hi = bounds() if region is None else (np.asarray(region[0]), np.asarray(region[1]))
    xs = np.arange(lo[0], hi[0], voxel, dtype=np.float32)
    ys = np.arange(lo[1], hi[1], voxel, dtype=np.float32)
    zs = np.arange(lo[2], hi[2], voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    add(vol, xs, ys, zs, (-PLINTH[0] - 3, -PLINTH[1] - 3, -1), (PLINTH[0] + 3, PLINTH[1] + 3, PL_H + 5),
        plinth_sdf, k=0, chunk=8)
    say(f"  plinth {time.time() - t0:.0f}s")
    for pts, rr in tubes:
        add_tube(vol, xs, ys, zs, pts, np.asarray(rr, float), k=0.35)
    say(f"  {len(tubes)} needles, stalks and petals {time.time() - t0:.0f}s")
    for dm in domes:
        f, (blo, bhi) = dome_sdf(*dm)
        add(vol, xs, ys, zs, blo, bhi, f, k=0.3, chunk=16)
    sub = lp.sub
    for p in pits:                                        # floret pits
        sl = sub(xs, ys, zs, p - 0.5, p + 0.5)
        if any(s_.stop <= s_.start for s_ in sl):
            continue
        X, Y, Z = np.meshgrid(xs[sl[0]], ys[sl[1]], zs[sl[2]], indexing="ij")
        vol[sl] = np.maximum(vol[sl], 0.3 - np.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2))
    say(f"  {len(domes)} flower heads {time.time() - t0:.0f}s")
    f, (blo, bhi), _ = goose
    add(vol, xs, ys, zs, blo, bhi, f, k=0.4, chunk=8)
    say(f"  nēnē {time.time() - t0:.0f}s")
    for pts, rr in stems:
        add(vol, xs, ys, zs, pts.min(0) - 2.5, pts.max(0) + 2.5,
            lambda x, y, z, s_=pts, r_=rr: capsule_chain(x, y, z, s_, r_), k=0.8)
    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])
    return vol, (xs[0], ys[0], zs[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.25, help="mesh resolution, mm")
    ap.add_argument("--preview", action="store_true", help="skip support planning")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    t0 = time.time()
    vol, origin = build(args.voxel)
    mesh = decimate(mesh_volume(vol, args.voxel, origin), 0.006)
    print(f"silversword: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, {time.time() - t0:.0f}s")
    out = os.path.join(args.outdir, "silversword.stl")
    mesh.export(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
