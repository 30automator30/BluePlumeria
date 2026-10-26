#!/usr/bin/env python3
"""Bird of paradise desk sculpture (resin / MSLA, Saturn 4 Ultra), about
120 mm tall, printed in one piece upright on its plinth with no supports.

  plinth   two stepped squares with chamfered edges, sized by phi.
  stem     the scape rises from a sheath in a flowing S-curve, a slim
           ribbon spiralling up it, and bends sharply into the spathe.
  spathe   the beak: a long boat-shaped bract with a keel below and a
           groove along the top, tilted up, its tip dipping slightly.
  crest    three tall orange sepals fan up out of the groove with the
           arrow-shaped blue tongue between them, and two younger sepals
           rise behind: the layered look of a real bloom.
  leaves   three paddle leaves on sweeping stalks: gently twisted blades
           with a raised midrib, parallel side veins and a waved margin.

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
PLINTH = (20.0, 20.0 / PHI ** 0.5 * 1.0, 6.0, 11.0)   # lower half-size, upper half-size, step z, top z
ROOT = np.array([-8.0, 0.0, PLINTH[3]])             # where the plant leaves the plinth
BEND = np.array([-9.0, 0.0, H / PHI + 2.0])          # scape bends into the spathe here
SPATHE_DIR = unit([np.cos(np.radians(18)), 0.0, np.sin(np.radians(18))])
SPATHE_L = 46.0


# --------------------------------------------------------------- blades

def blade_sdf(fr, L, W, T, shape="lance", twist=0.0, bend=0.0, fold=0.04,
              rib=0.35, veins=None, wave=0.0):
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
        return W * np.clip(tc / 0.1, 0, 1) ** 0.5 * np.clip((1 - tc) / 0.3, 0, 1) ** 0.65

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
        return d
    return f, box(fr, (-1, -W - 3, -L * abs(bend) - 4), (L + 1, W + 3, L * abs(bend) + 4))


# --------------------------------------------------------------- spathe

def spathe_sdf():
    """The beak: boat-shaped, keeled below, grooved along the top."""
    fr = frame(BEND, SPATHE_DIR, UP)
    L = SPATHE_L
    tt = np.linspace(0, 1, 801)
    nrm = (tt ** 0.35 * (1 - tt) ** 0.75).max()

    def f(x, y, z):
        u, v, w = local(fr, x, y, z)
        t = np.clip(u / L, 0, 1)
        prof = t ** 0.35 * (1 - t) ** 0.75 / nrm
        a = 4.3 * prof + 0.2                                  # half-width
        b = 3.3 * prof * (1 - 0.15 * t) + 0.2                 # half-height
        c = -2.6 * t ** 2                                     # the tip dips
        wl = w - c
        bb = np.where(wl < 0, b * 1.3, b)                     # deeper keel side
        r = np.sqrt((v / a) ** 2 + (wl / bb) ** 2)
        d = (r - 1) * np.minimum(a, bb)
        d = np.maximum(d, np.maximum(-u, u - L))
        keel = np.maximum(np.abs(v) - 0.5, np.abs(wl + bb * 0.95) - 0.5)
        d = smin(d, np.maximum(keel, np.maximum(-u + 2, u - L + 3)), 0.6)
        groove = np.maximum(np.sqrt(v ** 2 + (wl - b) ** 2) - 0.9,
                            np.maximum(0.12 * L - u, u - 0.7 * L))
        return np.maximum(d, -groove)
    return f, box(fr, (-3, -7, -9), (L + 2, 7, 7)), fr


def spathe_top(t):
    """World point on the spathe's top groove at fraction t of its length."""
    tt = np.linspace(0, 1, 801)
    nrm = (tt ** 0.35 * (1 - tt) ** 0.75).max()
    prof = t ** 0.35 * (1 - t) ** 0.75 / nrm
    b = 3.3 * prof * (1 - 0.15 * t) + 0.2
    w = -2.6 * t ** 2 + b - 0.6
    fr = frame(BEND, SPATHE_DIR, UP)
    return BEND + t * SPATHE_L * fr[1] + w * fr[3]


# ----------------------------------------------------------------- parts

def parts():
    tubes, things = [], []
    # sheath at the base
    zs = np.linspace(ROOT[2] - 1, ROOT[2] + 12, 20)
    tubes.append((np.column_stack([ROOT[0] + 0 * zs, ROOT[1] + 0 * zs, zs]),
                  4.2 - 1.6 * (zs - zs[0]) / (zs[-1] - zs[0])))
    # scape: an S-curve up to the bend, then a knuckle into the spathe
    scape = bezier(ROOT + [0, 0, 6], ROOT + [7, 0, 30], BEND + [-11, -2, -24], BEND, n=60)
    tubes.append((scape, np.linspace(2.4, 2.0, 60)))
    tubes.append((np.array([BEND, BEND + 3 * SPATHE_DIR]), np.array([2.9, 2.6])))
    # ribbon spiralling up the scape (stylised)
    s = np.linspace(0, 1, 140)
    idx = (s * (len(scape) - 1)).astype(int)
    tan = np.gradient(scape, axis=0)[idx]
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    e1 = np.cross(tan, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(tan, e1)
    ang = 2 * np.pi * 1.25 * s
    rib = scape[idx] + 2.7 * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
    tubes.append((rib[4:-6], np.linspace(1.05, 0.8, len(rib) - 10)))

    # crest: three sepals and the tongue from the groove, two younger behind
    side = np.array([0, 1.0, 0])
    p = spathe_top(0.42)
    for L, el, yaw in ((40, 52, 4), (37, 64, -10), (33, 77, 12)):
        d = unit(np.cos(np.radians(el)) * np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0])
                 + np.sin(np.radians(el)) * UP)
        things.append(blade_sdf(frame(p, d, side), L, 2.1, 1.2, "lance", twist=0.25,
                                bend=-0.06, fold=0.12, rib=0.3, veins="long"))
    d = unit(np.cos(np.radians(58)) * np.array([1.0, 0.12, 0]) + np.sin(np.radians(58)) * UP)
    things.append(blade_sdf(frame(p + [1.0, 0.8, 0], d, side), 31, 1.9, 1.1, "lance", twist=0.1,
                            fold=0.0, rib=0.25))
    tip = p + np.array([1.0, 0.8, 0]) + 31 * d * 0.86
    for sg in (-1, 1):                                    # arrowhead lobes, swept up
        ld = unit(d + sg * 0.9 * np.array([0, 1.0, 0]) - 0.2 * np.array([1.0, 0, 0]))
        tubes.append((np.array([tip, tip + 3.0 * ld]), np.array([0.7, 0.4])))
    q = spathe_top(0.24)
    for L, el, yaw in ((27, 88, 6), (23, 100, -8)):
        d = unit(np.cos(np.radians(el)) * np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0])
                 + np.sin(np.radians(el)) * UP)
        things.append(blade_sdf(frame(q, d, side), L, 1.9, 1.1, "lance", twist=0.2,
                                fold=0.1, rib=0.25, veins="long"))

    # leaves: sweeping stalks, gently twisted paddle blades
    for base, top, lean, L, W, tw in (
            (ROOT + [-1, 1.5, 9], np.array([-28.0, 9.0, 40.0]), [-0.35, 0.12, 1.0], 44, 9.0, 0.45),
            (ROOT + [1, -1.5, 8], np.array([14.0, -12.0, 32.0]), [0.30, -0.25, 1.0], 38, 8.0, -0.4),
            (ROOT + [-1, -1, 10], np.array([-17.0, -16.0, 50.0]), [-0.2, -0.3, 1.0], 34, 7.5, 0.35)):
        lean = unit(lean)
        swing = unit([top[0] - base[0], top[1] - base[1], 0])
        stalk = bezier(base, base + np.array([0, 0, 14.0]) - 5 * swing, top - 13 * lean + 3 * swing, top, n=40)
        tubes.append((stalk, np.linspace(1.6, 1.15, 40)))
        out = unit([top[0] - ROOT[0], top[1] - ROOT[1], 0])
        things.append(blade_sdf(frame(top - 1.0 * lean, lean, out), L, W, 1.1, "leaf", twist=tw,
                                bend=0.04, fold=0.05, rib=0.4, veins="parallel", wave=0.25))
    return tubes, things


def build(voxel, stems=(), quiet=False):
    say = (lambda *a: None) if quiet else print
    t0 = time.time()
    xs = np.arange(-46, 46, voxel, dtype=np.float32)
    ys = np.arange(-34, 30, voxel, dtype=np.float32)
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
    f, (blo, bhi), _ = spathe_sdf()
    add(vol, xs, ys, zs, blo, bhi, f, k=1.0, chunk=8)
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
