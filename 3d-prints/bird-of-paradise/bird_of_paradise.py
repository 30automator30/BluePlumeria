#!/usr/bin/env python3
"""Tropical arrangement (resin / MSLA, Saturn 4 Ultra), about 120 mm tall,
printed in one piece upright on its plinth with no supports: a bird of
paradise clump with a pride of Barbados spike.

  plinth   two stepped squares with chamfered edges, sized by phi.
  clump    leaf sheaths twisting round each other; seven tall paddle
           leaves on swaying S-curved stalks fan up out of it, curving in
           and twisting like flames, split along their side veins like the giant
           bird of paradise, with a raised midrib and parallel veins.
  blooms   three large bird of paradise heads at staggered heights, each
           facing its own way, in full detail: a sheathing bract on the
           scape; a sharp-keeled, upturned beak with colour-band guide
           lines; three crest sepals with a keel, a channel and drawn-out
           tips, pale throat bracts; the blue arrowhead tongue with its
           notch and stigma whisker; a younger crest and tongue behind.
  curls    four golden-spiral tendrils curl out from the base, as on the
           plumeria sculpture.
  pride    a pride of Barbados spike at the back: five frilly, cupped
           flowers set at the golden angle, each with seven long stamens
           arching out and up, and buds at the tip.

Everything rises from where it attaches; whatever would still start in
mid-air gets a slim stem from the same planner as the hibiscus bonsai.

Usage
  python3 bird_of_paradise.py --voxel 0.25 --preview     # quick look
  python3 bird_of_paradise.py                            # final STL
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("hibiscus-bonsai", "fractal-plumeria", "plumeria-ring"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import lattice_plumeria as lp  # noqa: E402
from bonsai_hibiscus import UP, box, clear_stems, frame, local, rrect, unit  # noqa: E402
from fractal_plumeria import capsule_chain, mesh_volume  # noqa: E402
from lattice_plumeria import PHI, BASE_CUT, add, bezier, soft_band  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402

H = 120.0
PLINTH = (21.0, 21.0 / PHI ** 0.5, 6.0, 11.0)       # lower half-size, upper half-size, step z, top z
ROOT = np.array([0.0, 0.0, PLINTH[3]])             # where the clump leaves the plinth

# three large bird of paradise heads: (angle round the clump, distance
# out, bend height, beak length, beak tilt up), staggered like a real clump
BLOOMS = [(20, 9, 90, 38, 24), (150, 14, 79, 36, 28), (290, 15, 68, 35, 26)]
# seven split leaves: (angle, lean from vertical, blade base height, length)
LEAVES = [(200, 42, 40, 44), (110, 36, 46, 46), (300, 40, 38, 42), (30, 46, 34, 40),
          (235, 26, 52, 48), (165, 50, 30, 38), (75, 32, 48, 44)]
PRIDE_ANGLE = 220                                    # pride of Barbados spike, between two heads


# --------------------------------------------------------------- blades

def blade_sdf(fr, L, W, T, shape="lance", twist=0.0, bend=0.0, fold=0.04,
              rib=0.35, veins=None, wave=0.0, splits=(), channel=0.0, vein_gap=2.6):
    """A thin blade in the frame fr (u along it, w its face normal):
    'lance' tapers to points at both ends, 'leaf' is an oblong paddle with
    a pointed tip. twist turns it about its length, bend curls it toward
    +w, fold gives a gentle V. 'acum' is a lance with a long drawn-out tip,
    'arrow' the bird of paradise tongue (an arrowhead with a side notch).
    A raised midrib and optional veins run on the +w face for painting, and
    channel cuts a groove down the middle of the -w face."""
    tt = np.linspace(0, 1, 801)
    tip = 1.4 if shape == "acum" else 0.9
    nrm = (tt ** 0.5 * (1 - tt) ** tip).max()

    def half_w(t):
        tc = np.clip(t, 0, 1)
        if shape in ("lance", "acum"):
            return W * tc ** 0.5 * (1 - tc) ** tip / nrm
        if shape == "arrow":
            base = W * tc ** 0.5 * (1 - tc) ** 0.9 / nrm
            # barbs widen gradually up to t = 0.82 (so they grow from below)
            # and drop away sharply above: the notch of the arrowhead
            ramp = np.clip((tc - 0.5) / 0.32, 0, 1) ** 1.5
            lobe = 1.2 * W * np.where(tc < 0.82, ramp, np.exp(-((tc - 0.82) / 0.035) ** 2))
            return base + lobe
        return W * np.clip(tc / 0.32, 0, 1) ** 0.8 * np.clip((1 - tc) / 0.3, 0, 1) ** 0.65   # tapered base

    def f(x, y, z):
        u, v, w = local(fr, x, y, z)
        t = u / L
        tc = np.clip(t, 0, 1)
        ph = twist * tc
        c, s = np.cos(ph), np.sin(ph)
        w = w - bend * L * tc ** 2
        v, w = v * c + w * s, -v * s + w * c
        half = half_w(t) + wave * np.sin(2 * np.pi * 7 * tc) * (tc > 0.05)
        o = np.maximum(np.abs(v) - half - 1e-3, np.maximum(-u, u - L))
        h = fold * v ** 2
        d = soft_band(o, np.abs(w - h) - T / 2, 2.0, T / 2)
        if rib:
            mr = np.maximum(np.abs(v) - 0.45, o + 0.8)
            d = smin(d, soft_band(mr, np.abs(w - h - T / 2) - rib, 0.45, rib), 0.3)
        if veins == "parallel":
            line = np.abs(((u - 0.55 * np.abs(v)) / vein_gap) % 1.0 - 0.5) * vein_gap * 0.88
            vn = np.maximum(line - 0.22, np.maximum(o + 0.7, 0.8 - np.abs(v)))
            d = smin(d, soft_band(vn, np.abs(w - h - T / 2) - 0.14, 0.22, 0.14), 0.2)
        elif veins == "long":
            sv = v / np.maximum(half, 1e-3)
            line = np.min([np.abs(sv - k) for k in (-0.5, 0.5)], axis=0) * half
            vn = np.maximum(line - 0.2, o + 0.6)
            d = smin(d, soft_band(vn, np.abs(w - h - T / 2) - 0.12, 0.2, 0.12), 0.2)
        if channel:
            ch = np.maximum(np.abs(v) - 0.3, o + 0.7)
            d = np.maximum(d, -soft_band(ch, np.abs(w - h + T / 2) - channel, 0.3, channel))
        for k, sd in splits:                                   # slots along the side veins
            line = np.abs(u - k * L - 0.55 * np.abs(v))
            slot = np.maximum(line - 0.7, np.maximum(0.3 * half - np.abs(v), -sd * v))
            d = np.maximum(d, -slot)
        return d
    return f, box(fr, (-1, -W - 3, -L * abs(bend) - 4), (L + 1, W + 3, L * abs(bend) + 4))


# --------------------------------------------------------------- spathe

_tt = np.linspace(0, 1, 801)
_NRM = (_tt ** 0.35 * (1 - _tt) ** 0.75).max()


def spathe_sdf(bend, dirn, L):
    """The beak: boat-shaped with a sharp keel below, a groove along the
    top and an upturned tip. Shallow guide lines mark the colour bands (a
    ring near the base, a line along each side above the keel)."""
    fr = frame(bend, dirn, UP)
    sc = L / 46.0

    def f(x, y, z):
        u, v, w = local(fr, x, y, z)
        t = np.clip(u / L, 0, 1)
        prof = t ** 0.35 * (1 - t) ** 0.75 / _NRM
        a = (4.3 * prof + 0.2) * sc + 0.3                     # half-width
        b = (3.3 * prof * (1 - 0.15 * t) + 0.2) * sc + 0.3    # half-height
        c = sc * (-2.2 * t ** 2 + 2.8 * t ** 6)               # dips, then the tip turns up
        wl = w - c
        bb = np.where(wl < 0, b * 1.3, b)                     # deeper keel side
        r = np.sqrt((v / a) ** 2 + (wl / bb) ** 2)
        d = (r - 1) * np.minimum(a, bb)
        d = np.maximum(d, np.maximum(-u, u - L))
        keel = np.maximum(np.abs(v) - 0.35, np.abs(wl + bb * 1.0) - 0.45)
        d = smin(d, np.maximum(keel, np.maximum(-u + 2, u - L + 3)), 0.35)
        ring = np.abs(u - 0.14 * L) - 0.16                      # colour-band guides
        side = np.abs(wl + 0.35 * bb) - 0.16
        guide = np.maximum(np.minimum(ring, np.maximum(side, np.maximum(0.1 * L - u, u - 0.9 * L))),
                           -d - 0.13)
        d = np.maximum(d, -guide)
        groove = np.maximum(np.sqrt(v ** 2 + (wl - b) ** 2) - 0.75,
                            np.maximum(0.12 * L - u, u - 0.7 * L))
        return np.maximum(d, -groove)
    return f, box(fr, (-3, -7, -9), (L + 2, 7, 7))


def spathe_top(bend, dirn, L, t):
    sc = L / 46.0
    prof = t ** 0.35 * (1 - t) ** 0.75 / _NRM
    b = (3.3 * prof * (1 - 0.15 * t) + 0.2) * sc + 0.3
    fr = frame(bend, dirn, UP)
    return bend + t * L * fr[1] + (sc * (-2.2 * t ** 2 + 2.8 * t ** 6) + b - 0.5) * fr[3]


def heading(yaw, el):
    return unit(np.cos(np.radians(el)) * np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0])
                + np.sin(np.radians(el)) * UP)


def sway(path, amp, phase=0.0):
    """Swing a path sideways in an S (zero at both ends), so stalks flow."""
    s = np.linspace(0, 1, len(path))[:, None]
    tan = np.gradient(path, axis=0)
    side = np.cross(tan, UP)
    side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
    return path + amp * np.sin(2 * np.pi * s + phase) * np.sin(np.pi * s) ** 1.5 * side   # ends keep their line


def bloom(tubes, things, ang, rad, zb, L, tilt):
    """One bird of paradise head in full detail: scape with a sheathing
    bract, keeled beak, three crest sepals (keel, channel, drawn-out tips)
    with pale throat bracts, the blue arrowhead tongue with its stigma
    whisker, and a second, younger crest and tongue behind. Head parts go
    in their own lists so the final build can mesh them more finely."""
    out = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0])
    bend = np.array([0, 0, zb]) + rad * out
    dirn = heading(ang, tilt)
    root = ROOT + [0, 0, 5] + 1.5 * out
    scape = bezier(root, root + [0, 0, 0.35 * (zb - root[2])] + 2 * out,
                   bend - [0, 0, 0.3 * (zb - root[2])] - 4 * out, bend, n=60)
    scape = sway(scape, 4.5 if int(ang) % 2 == 0 else -4.5)
    tubes.append((scape, np.linspace(1.9, 1.6, 60)))
    ht, hb = [], []                                    # head tubes and blades
    ht.append((np.array([bend, bend + 2 * dirn]), np.array([2.3, 2.1])))
    hb.append(spathe_sdf(bend, dirn, L))
    # sheathing bract wrapping the scape just below the beak
    j = len(scape) - 14
    tg = unit(scape[j + 4] - scape[j])
    hb.append(blade_sdf(frame(scape[j], tg, out), 11.0, 2.4, 0.9, "acum", fold=0.32, rib=0.2))
    side = np.array([-out[1], out[0], 0])
    k = L / 38.0
    p = spathe_top(bend, dirn, L, 0.42)
    # pale throat bracts where the crest leaves the beak
    for sg in (-1, 1):
        hb.append(blade_sdf(frame(p - 0.4 * dirn, heading(ang + 30 * sg, 38), side), 6.5 * k, 1.5,
                            0.8, "lance", fold=0.3, rib=0.0))
    for (Ls, el, yaw), bd in zip(((34, 50, 4), (31, 63, -10), (28, 76, 12)), (-0.12, 0.1, -0.08)):
        d = heading(ang + yaw, el)
        hb.append(blade_sdf(frame(p, d, side), Ls * k, 2.4, 1.1, "acum", twist=0.45, bend=bd,
                            fold=0.14, rib=0.25, channel=0.16))
    # the blue tongue: arrowhead with a notch, stigma whisker past its tip
    d = heading(ang + 6, 57)
    Lt = 27 * k
    hb.append(blade_sdf(frame(p + 0.6 * out, d, side), Lt, 1.5, 1.0, "arrow", twist=0.1,
                        fold=0.05, rib=0.2))
    tip = p + 0.6 * out + Lt * d
    ht.append((np.array([tip - 1.0 * d, tip + 4.5 * heading(ang + 6, 62)]), np.array([0.45, 0.36])))
    # a second, younger crest and tongue rising behind
    q = spathe_top(bend, dirn, L, 0.25)
    for Ls, el, yaw in ((24, 86, 6), (21, 98, -8)):
        d = heading(ang + yaw, el)
        hb.append(blade_sdf(frame(q, d, side), Ls * k, 2.0, 1.0, "acum", twist=0.25, fold=0.12,
                            rib=0.2, channel=0.12))
    d = heading(ang - 4, 80)
    hb.append(blade_sdf(frame(q + 0.5 * out, d, side), 19 * k, 1.3, 0.9, "arrow", fold=0.05, rib=0.15))
    tubes.extend(ht)
    things.extend(hb)
    lo = np.min([np.min(t_[0], 0) for t_ in ht] + [b_[1][0] for b_ in hb], axis=0) - 1
    hi = np.max([np.max(t_[0], 0) for t_ in ht] + [b_[1][1] for b_ in hb], axis=0) + 1
    return ht, hb, (lo, hi)


# ------------------------------------------------------- pride of Barbados

def pride_flower(fr, R=8.0):
    """Five frilly clawed petals cupped up, seven long stamens arching out
    and up with anther tips."""
    stam = []
    for k in range(7):
        a = 2 * np.pi * (k - 3) / 14 + 0.2                    # a fan to one side, like the real flower
        o = np.array([np.cos(a), np.sin(a), 0])
        stam.append(bezier(np.array([0, 0, 0.6]), 1.5 * o + [0, 0, 4.0],
                           (6.0 + k % 3) * o + [0, 0, 7.5], (8.5 + k % 3) * o + [0, 0, 11.0 + k % 2], n=16))

    def f(x, y, z):
        lx, ly, lz = local(fr, x, y, z)
        rho = np.sqrt(lx ** 2 + ly ** 2) + 1e-6
        ang = np.arctan2(ly, lx)
        d = np.full(lx.shape, 1e9, np.float32)
        for i in range(5):
            dphi = np.angle(np.exp(1j * (ang - 2 * np.pi * i / 5)))
            v = rho * dphi
            Ri = R * (0.75 if i == 0 else 1.0)                 # the smaller, upright standard petal
            t = np.clip(rho / Ri, 0, 1)
            half = 4.2 * np.clip((t - 0.15) / 0.5, 0.18, 1) ** 0.8 * np.sqrt(np.clip(1 - ((t - 0.62) / 0.4) ** 2, 0, 1)) + 0.4
            frill = 0.35 * np.sin(11 * np.arctan2(v, rho - 0.3 * R)) * np.clip((t - 0.4) / 0.4, 0, 1)
            o = np.maximum(np.abs(v) - half * (0.8 if i == 0 else 1.0), rho - Ri) + frill
            h = (5.0 if i == 0 else 3.0) * t ** 1.3            # cupped up; the standard stands taller
            d = smin(d, soft_band(o, np.abs(lz - h) - 0.45, 2.0, 0.45), 0.3)
        d = smin(d, np.sqrt(lx ** 2 + ly ** 2 + (lz + 0.8) ** 2) - 1.5, 0.6)
        for st in stam:
            d = smin(d, capsule_chain(lx, ly, lz, st, np.linspace(0.42, 0.36, len(st))), 0.25)
            e = st[-1]
            d = smin(d, np.sqrt((lx - e[0]) ** 2 + (ly - e[1]) ** 2 + (lz - e[2] - 0.2) ** 2) - 0.5, 0.25)
        return d
    return f, box(fr, (-15, -15, -3), (15, 15, 12))


def pride_spike(tubes, things):
    out = np.array([np.cos(np.radians(PRIDE_ANGLE)), np.sin(np.radians(PRIDE_ANGLE)), 0])
    top = 17 * out + np.array([0, 0, 104.0])
    stem = bezier(ROOT + [0, 0, 4] + 2 * out, ROOT + [0, 0, 40] + 4 * out, top - [0, 0, 30] - 1 * out, top, n=60)
    tubes.append((stem, np.linspace(1.7, 1.1, 60)))
    for k in range(5):
        j = int((0.62 + 0.07 * k) * (len(stem) - 1))
        a = np.radians(PRIDE_ANGLE) + k * GOLDEN
        o = np.array([np.cos(a), np.sin(a), 0])
        tip = stem[j] + 5.0 * o + [0, 0, 3.5]
        tubes.append((np.array([stem[j], tip]), np.array([0.8, 0.7])))
        axis = unit(0.2 * o + UP)
        fx = unit(o - (o @ axis) * axis)
        things.append(pride_flower((tip + 0.6 * axis, fx, np.cross(axis, fx), axis)))
    for k in range(3):                                         # buds at the tip
        a = np.radians(PRIDE_ANGLE) + 2.1 * k
        o = np.array([np.cos(a), np.sin(a), 0])
        b = top + 2.5 * o + [0, 0, 2.0 + 1.5 * k]
        tubes.append((np.array([top, b]), np.array([0.7, 0.6])))
        tubes.append((np.array([b, b + [0, 0, 1.0]]), np.array([1.9, 1.7])))    # round bud


GOLDEN = np.pi * (3 - np.sqrt(5))


# ----------------------------------------------------------------- parts

def parts():
    tubes, things = [], []
    # a fan of leaf sheaths at the base of the clump
    for k in range(5):
        a = 2 * np.pi * k / 5
        o = np.array([np.cos(a), np.sin(a), 0])
        ss = np.linspace(0, 1, 24)                     # sheaths twist round each other
        aa = a + 0.9 * ss
        rr = 2.6 - 2.0 * ss
        sh = np.column_stack([rr * np.cos(aa), rr * np.sin(aa), ROOT[2] - 0.5 + 16 * ss])
        tubes.append((sh, np.linspace(3.2, 1.6, 24)))
    # golden-spiral tendrils curling out from the base, as on the plumeria
    for k in range(4):
        a = np.radians(45 + 90 * k + 12 * (k % 2))
        o = np.array([np.cos(a), np.sin(a), 0])
        p0 = ROOT + [0, 0, 6] + 2.2 * o
        p1 = (17.0 + 3 * (k % 2)) * o + [0, 0, ROOT[2] + 20 + 7 * (k % 3)]
        arc = sway(bezier(p0, p0 + [0, 0, 9] + 2 * o, p1 - 7 * o - [0, 0, 2], p1, n=40), 2.5 * (-1) ** k)
        curl, _, _ = lp.golden_curl(arc[-1], arc[-1] - arc[-3], 4.2)
        tubes.append((np.vstack([arc, curl]),
                      np.concatenate([np.linspace(1.5, 1.1, len(arc)), np.linspace(1.1, 0.65, len(curl))])))
    heads = [bloom(tubes, things, *b_) for b_ in BLOOMS]
    pride_spike(tubes, things)
    # split paddle leaves on long stalks fanning up out of the clump
    rng = np.random.default_rng(3)
    for ang, lean, zb, L in LEAVES:
        out = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0])
        d = heading(ang, 90 - lean)
        base = ROOT + [0, 0, 8] + 1.5 * out
        top = np.array([0, 0, zb]) + (zb - ROOT[2]) * np.tan(np.radians(lean)) * 0.55 * out
        stalk = bezier(base, base + [0, 0, 0.35 * (zb - base[2])], top - 0.3 * (zb - base[2]) * d, top, n=40)
        stalk = sway(stalk, 5.0 * (-1) ** int(ang // 45))
        tubes.append((stalk, np.linspace(1.4, 1.0, 40)))
        n_split = 3 + int(rng.integers(0, 2))
        splits = [(float(f_), float(sd)) for f_, sd in zip(np.sort(rng.uniform(0.3, 0.85, n_split)),
                                                           rng.choice([-1, 1], n_split))]
        # face up and inward, so the gentle fold and curl lift the edges and tip
        things.append(blade_sdf(frame(stalk[-1] - 0.8 * d, d, -out), L, 10.0, 1.0, "leaf", twist=0.5,
                                bend=0.1, fold=0.03, rib=0.35, veins="parallel", wave=0.2, vein_gap=1.8,
                                splits=splits))
    return tubes, things, heads


def build(voxel, stems=(), quiet=False, region=None):
    """Distance field of the whole piece, or of just the box region
    (lo, hi), e.g. one head meshed finely for the final build."""
    say = (lambda *a: None) if quiet else print
    t0 = time.time()
    lo_, hi_ = region if region is not None else ((-50, -50, -0.6), (50, 50, H + 4))
    xs = np.arange(lo_[0], hi_[0], voxel, dtype=np.float32)
    ys = np.arange(lo_[1], hi_[1], voxel, dtype=np.float32)
    zs = np.arange(lo_[2], hi_[2], voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    lo, up, zstep, ztop = PLINTH

    def plinth(x, y, z):
        ch = np.clip(z - (zstep - 1.0), 0, 1.0)                      # chamfered top edges
        low = np.maximum(rrect(x, y, lo - ch, lo - ch, 3.0), np.maximum(-z, z - zstep))
        ch2 = np.clip(z - (ztop - 1.0), 0, 1.0)
        top = np.maximum(rrect(x, y, up - ch2, up - ch2, 2.0), np.maximum(-z, z - ztop))
        return np.minimum(low, top)
    add(vol, xs, ys, zs, (-lo - 1, -lo - 1, -1), (lo + 1, lo + 1, ztop + 1), plinth, k=0)

    tubes, things, heads = parts()
    for pts, rr in tubes:
        rr = np.broadcast_to(rr, (len(pts),))
        for k0 in range(0, len(pts) - 1, 10):
            seg, rad = pts[k0:k0 + 11], rr[k0:k0 + 11]
            pad = rad.max() + 1.0
            add(vol, xs, ys, zs, seg.min(0) - pad, seg.max(0) + pad,
                lambda x, y, z, s_=seg, r_=rad: capsule_chain(x, y, z, s_, r_), k=0.8)
    for f, (blo, bhi) in things:
        add(vol, xs, ys, zs, blo, bhi, f, k=0.6, chunk=8)
    say(f"  built {time.time() - t0:.0f}s")
    for pts, rr in stems:
        add(vol, xs, ys, zs, pts.min(0) - 2.5, pts.max(0) + 2.5,
            lambda x, y, z, s_=pts, r_=rr: capsule_chain(x, y, z, s_, r_), k=0.8)
    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])
    return vol, (xs[0], ys[0], zs[0])


def plan_stems(voxel=0.3, passes=8):
    stems = []
    for i in range(passes):
        vol, origin = build(voxel, stems, quiet=True)
        isl = lp.find_islands(vol, voxel, origin)
        print(f"  stem pass {i + 1}: {len(isl)} island(s), {len(stems)} stem(s)")
        if not isl:
            break
        stems += clear_stems(isl, vol, voxel, origin)
    return stems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.12, help="mesh resolution, mm")
    ap.add_argument("--preview", action="store_true", help="skip support planning")
    ap.add_argument("--out", default=os.path.join(HERE, "bird_of_paradise.stl"))
    args = ap.parse_args()
    t0 = time.time()
    vol, origin = build(args.voxel, () if args.preview else plan_stems())
    mesh = decimate(mesh_volume(vol, args.voxel, origin), 0.006)
    print(f"{len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, "
          f"{time.time() - t0:.0f}s")
    mesh.export(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
