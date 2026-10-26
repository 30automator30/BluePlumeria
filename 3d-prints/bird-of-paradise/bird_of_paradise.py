#!/usr/bin/env python3
"""Tropical arrangement (resin / MSLA, Saturn 4 Ultra), about 120 mm tall,
printed in one piece upright on its plinth with no supports: a bird of
paradise clump with a pride of Barbados spike.

  plinth   two stepped squares with chamfered edges, sized by phi.
  clump    a fan of leaf sheaths; seven tall paddle leaves on long stalks
           fan up out of it, split along their side veins like the giant
           bird of paradise, with a raised midrib and parallel veins.
  blooms   five bird of paradise heads at staggered heights, each facing
           its own way: a keeled, grooved beak on its scape, three crest
           sepals and the blue tongue, two younger sepals behind.
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

# five bird of paradise heads: (angle round the clump, distance out, bend
# height, beak length, beak tilt up), staggered like a real clump
BLOOMS = [(10, 8, 98, 30, 22), (150, 13, 90, 28, 26), (250, 14, 82, 27, 26),
          (60, 16, 74, 26, 28), (320, 16, 67, 25, 26)]
# seven split leaves: (angle, lean from vertical, blade base height, length)
LEAVES = [(200, 42, 40, 44), (110, 36, 46, 46), (300, 40, 38, 42), (30, 46, 34, 40),
          (235, 26, 52, 48), (165, 50, 30, 38), (75, 32, 48, 44)]
PRIDE_ANGLE = 285                                    # pride of Barbados spike, at the back


# --------------------------------------------------------------- blades

def blade_sdf(fr, L, W, T, shape="lance", twist=0.0, bend=0.0, fold=0.04,
              rib=0.35, veins=None, wave=0.0, splits=()):
    """A thin blade in the frame fr (u along it, w its face normal):
    'lance' tapers to points at both ends, 'leaf' is an oblong paddle with
    a pointed tip. twist turns it about its length, bend curls it toward
    +w, fold gives a gentle V. A raised midrib and optional veins run on
    the +w face for painting."""
    if shape == "lance":
        tt = np.linspace(0, 1, 801)
        nrm = (tt ** 0.5 * (1 - tt) ** 0.9).max()

    def half_w(t):
        tc = np.clip(t, 0, 1)
        if shape == "lance":
            return W * tc ** 0.5 * (1 - tc) ** 0.9 / nrm
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
            line = np.abs(((u - 0.55 * np.abs(v)) / 2.6) % 1.0 - 0.5) * 2.6 * 0.88
            vn = np.maximum(line - 0.22, np.maximum(o + 0.7, 0.8 - np.abs(v)))
            d = smin(d, soft_band(vn, np.abs(w - h - T / 2) - 0.14, 0.22, 0.14), 0.2)
        elif veins == "long":
            sv = v / np.maximum(half, 1e-3)
            line = np.min([np.abs(sv - k) for k in (-0.5, 0.5)], axis=0) * half
            vn = np.maximum(line - 0.2, o + 0.6)
            d = smin(d, soft_band(vn, np.abs(w - h - T / 2) - 0.12, 0.2, 0.12), 0.2)
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
    """The beak: boat-shaped, keeled below, grooved along the top."""
    fr = frame(bend, dirn, UP)
    sc = L / 46.0

    def f(x, y, z):
        u, v, w = local(fr, x, y, z)
        t = np.clip(u / L, 0, 1)
        prof = t ** 0.35 * (1 - t) ** 0.75 / _NRM
        a = (4.3 * prof + 0.2) * sc + 0.3                     # half-width
        b = (3.3 * prof * (1 - 0.15 * t) + 0.2) * sc + 0.3    # half-height
        c = -2.6 * sc * t ** 2                                # the tip dips
        wl = w - c
        bb = np.where(wl < 0, b * 1.3, b)                     # deeper keel side
        r = np.sqrt((v / a) ** 2 + (wl / bb) ** 2)
        d = (r - 1) * np.minimum(a, bb)
        d = np.maximum(d, np.maximum(-u, u - L))
        keel = np.maximum(np.abs(v) - 0.45, np.abs(wl + bb * 0.95) - 0.45)
        d = smin(d, np.maximum(keel, np.maximum(-u + 2, u - L + 3)), 0.5)
        groove = np.maximum(np.sqrt(v ** 2 + (wl - b) ** 2) - 0.75,
                            np.maximum(0.12 * L - u, u - 0.7 * L))
        return np.maximum(d, -groove)
    return f, box(fr, (-3, -7, -9), (L + 2, 7, 7))


def spathe_top(bend, dirn, L, t):
    sc = L / 46.0
    prof = t ** 0.35 * (1 - t) ** 0.75 / _NRM
    b = (3.3 * prof * (1 - 0.15 * t) + 0.2) * sc + 0.3
    fr = frame(bend, dirn, UP)
    return bend + t * L * fr[1] + (-2.6 * sc * t ** 2 + b - 0.5) * fr[3]


def heading(yaw, el):
    return unit(np.cos(np.radians(el)) * np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0])
                + np.sin(np.radians(el)) * UP)


def bloom(tubes, things, ang, rad, zb, L, tilt):
    """One bird of paradise head on its scape: beak facing out at angle
    ang, crest of three sepals and the blue tongue, two younger behind."""
    out = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0])
    bend = np.array([0, 0, zb]) + rad * out
    dirn = heading(ang, tilt)
    root = ROOT + [0, 0, 5] + 1.5 * out
    scape = bezier(root, root + [0, 0, 0.35 * (zb - root[2])] + 2 * out,
                   bend - [0, 0, 0.3 * (zb - root[2])] - 4 * out, bend, n=50)
    tubes.append((scape, np.linspace(1.8, 1.45, 50)))
    tubes.append((np.array([bend, bend + 2 * dirn]), np.array([2.1, 1.9])))
    things.append(spathe_sdf(bend, dirn, L))
    side = np.array([-out[1], out[0], 0])
    k = L / 46.0
    p = spathe_top(bend, dirn, L, 0.42)
    for Ls, el, yaw in ((40, 52, 4), (37, 64, -10), (33, 77, 12)):
        d = heading(ang + yaw, el)
        things.append(blade_sdf(frame(p, d, side), Ls * k * 1.15, 1.7, 1.1, "lance", twist=0.25,
                                bend=-0.05, fold=0.12, rib=0.25, veins="long"))
    d = heading(ang + 6, 58)
    things.append(blade_sdf(frame(p + 0.6 * out, d, side), 31 * k * 1.15, 1.5, 1.0, "lance",
                            twist=0.1, fold=0.0, rib=0.2))
    q = spathe_top(bend, dirn, L, 0.24)
    for Ls, el, yaw in ((27, 88, 6), (23, 100, -8)):
        d = heading(ang + yaw, el)
        things.append(blade_sdf(frame(q, d, side), Ls * k * 1.15, 1.5, 1.0, "lance", twist=0.2,
                                fold=0.1, rib=0.2))


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
            t = np.clip(rho / R, 0, 1)
            half = 4.2 * np.clip((t - 0.15) / 0.5, 0.18, 1) ** 0.8 * np.sqrt(np.clip(1 - ((t - 0.62) / 0.4) ** 2, 0, 1)) + 0.4
            frill = 0.35 * np.sin(11 * np.arctan2(v, rho - 0.3 * R)) * np.clip((t - 0.4) / 0.4, 0, 1)
            o = np.maximum(np.abs(v) - half, rho - R) + frill
            h = 3.0 * t ** 1.3                                  # cupped up
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
        tubes.append((np.array([b, b + [0, 0, 1.2]]), np.array([1.6, 1.4])))


GOLDEN = np.pi * (3 - np.sqrt(5))


# ----------------------------------------------------------------- parts

def parts():
    tubes, things = [], []
    # a fan of leaf sheaths at the base of the clump
    for k in range(5):
        a = 2 * np.pi * k / 5
        o = np.array([np.cos(a), np.sin(a), 0])
        sh = bezier(ROOT - [0, 0, 0.5] + 2.5 * o, ROOT + [0, 0, 5] + 2.0 * o,
                    ROOT + [0, 0, 10] + 1.0 * o, ROOT + [0, 0, 15] + 0.5 * o, n=16)
        tubes.append((sh, np.linspace(3.2, 1.6, 16)))
    for b_ in BLOOMS:
        bloom(tubes, things, *b_)
    pride_spike(tubes, things)
    # split paddle leaves on long stalks fanning up out of the clump
    rng = np.random.default_rng(3)
    for ang, lean, zb, L in LEAVES:
        out = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0])
        d = heading(ang, 90 - lean)
        base = ROOT + [0, 0, 8] + 1.5 * out
        top = np.array([0, 0, zb]) + (zb - ROOT[2]) * np.tan(np.radians(lean)) * 0.55 * out
        stalk = bezier(base, base + [0, 0, 0.35 * (zb - base[2])], top - 0.3 * (zb - base[2]) * d, top, n=40)
        tubes.append((stalk, np.linspace(1.4, 1.0, 40)))
        n_split = 3 + int(rng.integers(0, 2))
        splits = [(float(f_), float(sd)) for f_, sd in zip(np.sort(rng.uniform(0.3, 0.85, n_split)),
                                                           rng.choice([-1, 1], n_split))]
        # face up and inward, so the gentle fold and curl lift the edges and tip
        things.append(blade_sdf(frame(top - 0.8 * d, d, -out), L, 10.0, 1.0, "leaf", twist=0.2,
                                bend=0.03, fold=0.03, rib=0.35, veins="parallel", wave=0.2,
                                splits=splits))
    return tubes, things


def build(voxel, stems=(), quiet=False):
    say = (lambda *a: None) if quiet else print
    t0 = time.time()
    xs = np.arange(-50, 50, voxel, dtype=np.float32)
    ys = np.arange(-50, 50, voxel, dtype=np.float32)
    zs = np.arange(-0.6, H + 4, voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    lo, up, zstep, ztop = PLINTH

    def plinth(x, y, z):
        ch = np.clip(z - (zstep - 1.0), 0, 1.0)                      # chamfered top edges
        low = np.maximum(rrect(x, y, lo - ch, lo - ch, 3.0), np.maximum(-z, z - zstep))
        ch2 = np.clip(z - (ztop - 1.0), 0, 1.0)
        top = np.maximum(rrect(x, y, up - ch2, up - ch2, 2.0), np.maximum(-z, z - ztop))
        return np.minimum(low, top)
    add(vol, xs, ys, zs, (-lo - 1, -lo - 1, -1), (lo + 1, lo + 1, ztop + 1), plinth, k=0)

    tubes, things = parts()
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
