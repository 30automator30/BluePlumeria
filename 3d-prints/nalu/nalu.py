#!/usr/bin/env python3
"""Nalu: a breaking wave (resin / MSLA, Saturn 4 Ultra), about 128 mm wide.
One part, printed straight on the plate:

  base      a low oval of rippled sand in front, swell lines behind, with
            a cowrie (leho) and a cone shell pressed into the sand and a
            lacy line of swash foam where the last wave ran up.
  wave      a curling wave that peels from left to right. At the left end
            it throws a full barrel (the end is a clean cross-section of
            the curl); toward the right the curl spills into a foam roll
            and the swell sinks into the sea.
  foam      a lace of holes along the throwing lip, and pitted foam along
            the crest, the spilling roll and the swash line.
  plumeria  one five-petal blossom riding the top of the lip.

Printability: the base and the wave rise from the plate. The lip of the
barrel hangs free and takes light tree supports from the sand below; the
barrel roof and every other face close in from walls that rise to them.
Texture is cut in, never raised, on any surface that faces down.

Usage
  python3 nalu.py --voxel 0.4 --preview          # quick look
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("fractal-plumeria", "plumeria-ring", "hibiscus-bonsai", "ohia-bonsai", "honu"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
from fractal_plumeria import mesh_volume  # noqa: E402
from lattice_plumeria import BASE_CUT, add  # noqa: E402
from ohia_bonsai import oval  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402
from honu import drop_specks, fill_voids, trim_crumbs  # noqa: E402

# ---------------------------------------------------------------- layout
BASE = (66.0, 47.0)            # half-axes of the base oval
RIM_H = 2.6                    # base height at the rim
HMAX = 84.0                    # height of the barrel's crest above the plate
X0, X1 = -60.0, 62.0           # the wave runs from its barrel end (left) to the shoulder (right)
FLOWER = dict(x=4.0, R=15.0, tilt=np.radians(40))
COWRIE = (38.0, -38.0, np.radians(-25))
CONE = (-22.0, -43.0, np.radians(15))
WALL = 2.6                     # wall of the hollow wave
TUNNELS = (-40.0, -12.0, 16.0)  # x of the vent tunnels under the back of the wave


def smax(a, b, k):
    return -smin(-a, -b, k)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------- wave cross-section
# Profiles in (y, z/H): y in mm (wave travels toward -y, the viewer), z as
# a fraction of the local wave height. Listed round the outline.
BARREL = np.array([
    (38, -0.05), (36, 0.0), (30, 0.06), (22, 0.20), (14, 0.42), (7, 0.67), (2, 0.86),
    (-4, 0.975), (-11, 1.0), (-18, 0.955), (-24, 0.85), (-28, 0.71), (-29.5, 0.57),
    (-28, 0.46), (-25.5, 0.41), (-23.2, 0.45), (-23.8, 0.56), (-22, 0.69), (-17, 0.78),
    (-11, 0.795), (-6.5, 0.74), (-4, 0.63), (-4, 0.50), (-6.5, 0.36), (-11.5, 0.22),
    (-18, 0.105), (-26, 0.035), (-36, 0.0), (-38, -0.05)])
LIP = slice(9, 17)             # outline points that make the throwing lip

SPILL = np.array([
    (38, -0.05), (36, 0.0), (30, 0.07), (22, 0.23), (14, 0.48), (7, 0.73), (1, 0.90),
    (-5, 0.985), (-11, 1.0), (-16, 0.96), (-20, 0.87), (-23, 0.74), (-25, 0.58),
    (-26.5, 0.42), (-27.5, 0.28), (-29, 0.17), (-32, 0.08), (-36, 0.0), (-38, -0.05)])


def height(x):
    """Wave height: the full barrel at the left, sinking into the sea to the right."""
    return HMAX * (1 - 0.86 * sstep(-28, X1 + 4, x)) * (1 - 0.06 * sstep(40, 62, -x))


def curl(x):
    """1 where the wave throws a barrel, 0 where it spills."""
    return 1 - sstep(-6, 34, x)


def chaikin(p, n=3):
    for _ in range(n):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], np.stack([q, r], 1).reshape(-1, 2), p[-1:]])
    return p


def outline(prof, x, H, lip=None):
    p = prof.astype(float).copy()
    k = 0.42 + 0.58 * H / HMAX                     # a lower wave is also a narrower one
    p[:, 0] *= k
    p[:, 1] *= H
    if lip is not None:                            # the lip throws a little unevenly along its length
        w = np.zeros(len(p))
        w[lip] = np.sin(np.linspace(0, np.pi, lip.stop - lip.start))
        p[:, 0] -= w * (1.3 * np.sin(x / 7.3) + 0.8 * np.sin(x / 3.1 + 1.0))
        p[:, 1] += w * (0.9 * np.sin(x / 5.2 + 2.0))
    return chaikin(np.vstack([p, p[:1]]))


def poly_sdf(py, pz, P):
    """Signed distance in 2D from points to a closed polygon P (m, 2)."""
    a, b = P[:-1], P[1:]
    best = np.full(py.shape, 1e9)
    inside = np.zeros(py.shape, bool)
    for (ay, az), (by, bz) in zip(a, b):
        ey, ez = by - ay, bz - az
        wy, wz = py - ay, pz - az
        h = np.clip((wy * ey + wz * ez) / (ey * ey + ez * ez + 1e-12), 0, 1)
        best = np.minimum(best, (wy - h * ey) ** 2 + (wz - h * ez) ** 2)
        c = ((az <= pz) & (bz > pz)) | ((bz <= pz) & (az > pz))
        xint = ay + (pz - az) * ey / (ez + 1e-12)
        inside ^= c & (py < xint)
    d = np.sqrt(best)
    return np.where(inside, -d, d)


def bend(x):
    """The wave is a crescent in plan: its barrel end swings toward the
    viewer, so from the front you look into the barrel."""
    return 7.0 - 14.0 * ((X1 - x) / (X1 - X0)) ** 2


def wave_slice(x, Y, Z):
    """Wave SDF on one x-slice."""
    H, c = height(x), curl(x)
    Y = Y - bend(x)
    d = None
    if c > 0.01:
        d = poly_sdf(Y, Z, outline(BARREL, x, H, LIP))
    if c < 0.99:
        ds = poly_sdf(Y, Z, outline(SPILL, x, H))
        d = ds if d is None else c * d + (1 - c) * ds
    # round off the barrel end and the right end where it meets the sea
    # the barrel end is not a flat cut: the lip throws on past the face
    xend = X0 - 9.0 * sstep(0.5 * H, 0.95 * H, Z) * sstep(-6, -22, Y) - 3.0 * sstep(0.3 * H, 0.9 * H, Z)
    d = smax(d, xend - x, 3.0)
    # cut-in flow lines, uneven in spacing and depth, on the back and the face
    ph = (x + 2.6 * np.sin(Z / 9.0 + Y / 12.0) + 1.4 * np.sin(Z / 3.7 - x / 9.0)) / 5.2
    g = 0.5 + 0.5 * np.cos(2 * np.pi * ph)
    amp = 0.18 + 0.14 * (0.5 + 0.5 * np.sin(x / 6.3 + Z / 17.0))
    face = np.clip(1 - np.abs(d) / 0.8, 0, 1)
    lipface = sstep(-14, -24, Y / (0.42 + 0.58 * H / HMAX)) * sstep(0.5 * H, 0.65 * H, Z) * c
    d = d + (amp + 0.25 * lipface) * g ** 8 * face * sstep(0.06 * H, 0.18 * H, Z)
    return d


def lace_holes():
    """Holes through the throwing lip: foam lace. (x, z, rx, rz), each an
    oval stretched along the lip, running along y through the lip only.
    Scattered, mixed in size, thicker toward the lip's edge."""
    rng = np.random.default_rng(7)
    out = []
    tries = 0
    while len(out) < 70 and tries < 4000:
        tries += 1
        x = rng.uniform(X0 - 4, 12)
        H, c = height(x), curl(x)
        if c < 0.7:
            continue
        zf = rng.uniform(0.55, 0.78)
        z = zf * H
        rz = rng.uniform(0.55, 1.05) * (1.15 - 0.6 * (zf - 0.55) / 0.23)
        rx = rz * rng.uniform(1.3, 2.3)
        if any(((x - a) / (rx + ra + 0.9)) ** 2 + ((z - b) / (rz + rb + 0.9)) ** 2 < 1 for a, b, ra, rb in out):
            continue                                   # keep a solid web between holes
        out.append((x, z, rx, rz))
    return out


def foam_pits(x, y, z, d, n, s=1.5, r=0.5, depth=0.45):
    """Bubbly foam: a jittered scatter of round pits cut into the surface
    along the normal; none where the surface overhangs."""
    q = np.stack([x / s, y / s + 0.5 * np.floor(x / s), z / s + 0.5 * np.floor(y / s)], -1)
    i = np.floor(q)
    h = np.sin(i[..., 0] * 12.9898 + i[..., 1] * 78.233 + i[..., 2] * 37.719) * 43758.5453
    j = h - np.floor(h)
    rr = r * (0.55 + 0.6 * j)                           # bubbles of mixed sizes
    f = q - i - 0.5
    along = np.einsum("...i,...i->...", f, n)
    cell = np.sqrt(np.maximum(np.einsum("...i,...i->...", f, f) - along ** 2, 0)) * s
    wgt = np.clip((n[..., 2] + 0.2) / 0.3, 0, 1)
    return np.maximum(d, np.minimum(rr - cell, d + depth * wgt))


def foam_zone(x, y, z):
    """Where the wave is foamy: weight 0..1."""
    H, c = height(x), curl(x)
    k = 0.42 + 0.58 * H / HMAX
    yw, y = y, y - bend(x)
    # the crest and lip of the barrel
    zc = z / np.maximum(H, 1e-3)
    lipzone = np.clip(1 - np.hypot((y / k + 13) / 9, (zc - 0.99) / 0.07), 0, 1) * c
    # the spilling roll
    roll = np.clip(1 - np.hypot((y / k + 24) / 8, (zc - 0.55) / 0.5), 0, 1) * (1 - c) * (H > 8)
    # the swash line on the sand in front
    sw = np.exp(-((yw - swash_y(x)) / 2.5) ** 2) * (z < 6)
    return np.maximum(np.maximum(lipzone, roll), sw)


# ------------------------------------------------------------------ base

def swash_y(x):
    return -47.0 + 2.2 * np.sin(x / 9.0) + 1.2 * np.sin(x / 4.1 + 1.3)


def base_top(x, y):
    """Sand in front (ripples, rising slightly to the waterline), swell lines
    behind the wave."""
    front = 1 - sstep(-38, -26, y)
    rip = 0.38 * np.sin(0.75 * (x + 2.0 * np.sin(y / 5.0))) * front
    swell = 0.30 * np.sin(0.45 * y + 0.6 * np.sin(x / 13.0)) * sstep(25, 34, y)
    # the swash edge: a soft scalloped step where the last wash stopped
    step = 0.45 * sstep(-0.6, 0.6, y - swash_y(x))
    return RIM_H + 1.1 + rip + swell + step


def base_sdf(x, y, z):
    flare = 1.4 * np.clip((RIM_H - z) / RIM_H, 0, 1)
    d = np.maximum(oval(x, y, BASE[0] + flare, BASE[1] + flare), z - base_top(x, y))
    return np.maximum(d, -z)


def ell(x, y, z, rx, ry, rz):
    k0 = np.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
    k1 = np.sqrt((x / rx ** 2) ** 2 + (y / ry ** 2) ** 2 + (z / rz ** 2) ** 2)
    return np.where(k1 > 1e-9, k0 * (k0 - 1) / np.maximum(k1, 1e-9), -min(rx, ry, rz))


def cowrie_sdf():
    cx, cy, a = COWRIE
    c = np.array([cx, cy, base_top(cx, cy) + 0.7])
    ca, sa = np.cos(a), np.sin(a)

    def f(x, y, z):
        u = (x - c[0]) * ca + (y - c[1]) * sa
        v = -(x - c[0]) * sa + (y - c[1]) * ca
        d = ell(u, v, z - c[2], 6.5, 4.2, 3.4)
        return d - 0.25 * np.cos(np.pi * u / 6.5) * np.exp(-(v / 1.4) ** 2)
    return f, (c - [8, 8, 4], c + [8, 8, 5])


def cone_sdf():
    """A cone shell lying on its side, its spire turned, half in the sand."""
    cx, cy, a = CONE
    c = np.array([cx, cy, base_top(cx, cy) + 0.9])
    ca, sa = np.cos(a), np.sin(a)
    L = 13.0

    def f(x, y, z):
        u = (x - c[0]) * ca + (y - c[1]) * sa + L / 2       # 0 at the spire, L at the base
        v = -(x - c[0]) * sa + (y - c[1]) * ca
        w = z - c[2]
        t = np.clip(u / L, 0, 1)
        r = 4.4 * np.where(t < 0.82, t / 0.82, 1 - 0.55 * (t - 0.82) / 0.18)
        rr = np.hypot(v, w)
        d = (rr - r) * 0.92
        d = np.maximum(d, np.maximum(-u, u - L))
        # spiral whorls on the spire and fine growth lines, cut in
        d = d + 0.18 * (0.5 + 0.5 * np.cos(2 * np.pi * u / 1.6)) * (t < 0.3)
        return d
    return f, (c - [10, 10, 5], c + [10, 10, 6])


# -------------------------------------------------------------- plumeria

def plumeria_sdf():
    """A five-petal plumeria resting on the top of the wave. Petals are
    obovate, twisted pinwheel-fashion, cupped and curling up at the tip;
    they all rise from the throat, so none hangs below it."""
    x0, R, tilt = FLOWER["x"], FLOWER["R"], FLOWER["tilt"]
    # sit it on the crest at x0: find the top of the wave in that slice
    ys = np.linspace(-40, 40, 321)
    zs = np.linspace(0, HMAX + 5, 361)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    d = wave_slice(x0, Y, Z)
    solid = d < 0
    ztop = np.where(solid.any(1), zs[np.argmax(solid[:, ::-1], 1) * -1 + len(zs) - 1], -1)
    j = int(np.argmax(ztop))
    yc = ys[j] + 2.0                                   # just behind the crest line
    zc = ztop[j] - 0.4
    C = np.array([x0, yc, zc])
    ct, st = np.cos(tilt), np.sin(tilt)
    Rm = np.array([[1, 0, 0], [0, ct, st], [0, -st, ct]])    # tip it toward the viewer

    def petal(u, v, w):
        t = np.clip(u / R, 0, 1)
        tt = np.clip(u / R, 0, 1.0)
        wid = 0.47 * R * np.minimum(tt / 0.62, 1) ** 0.8 * np.clip(1 - (np.maximum(tt - 0.62, 0) / 0.38) ** 2.2, 0, 1)
        wid = wid * np.where(v > 0, 1.1, 0.9) + 1e-3
        outl = np.maximum(np.abs(v) - wid, np.maximum(-u, u - R))
        vn = np.clip(v / wid, -1, 1)
        tip = np.maximum(t - 0.55, 0) / 0.45
        mid = (3.2 * np.sin(np.pi * 0.5 * np.minimum(t * 1.5, 1)) + 0.9 * vn * t
               + 0.8 * vn * vn * t + 1.6 * tip * tip)
        thick = 1.35 + 0.6 * (1 - t)
        shell = np.abs(w - mid) - thick / 2
        return smax(outl, shell, 0.3)

    def f(x, y, z):
        p = np.stack([x - C[0], y - C[1], z - C[2]], -1) @ Rm
        u0, v0, w0 = p[..., 0], p[..., 1], p[..., 2]
        d = np.full(x.shape, 1e9)
        for k in range(5):
            a = 2 * np.pi * k / 5 + 0.3
            off = 0.15 * R
            cx_, cy_ = -off * np.sin(a), off * np.cos(a)
            u = (u0 - cx_) * np.cos(a) + (v0 - cy_) * np.sin(a)
            v = -(u0 - cx_) * np.sin(a) + (v0 - cy_) * np.cos(a)
            d = np.minimum(d, petal(u, v, w0))
        # the throat: a low dome with a small funnel, and a stalk into the wave
        rr = np.hypot(u0, v0)
        dome = np.sqrt(rr ** 2 + (np.maximum(w0 - 1.2, 0) * 1.5) ** 2) - 0.22 * R
        dome = np.maximum(dome, -w0 - 3.0)
        fun = np.sqrt(rr ** 2 + (w0 - 1.2 - 0.6 * 0.22 * R) ** 2) - 0.08 * R
        d = smin(d, smax(dome, -fun, 0.2), 0.6)
        return d
    lo, hi = C - [R + 7, R + 7, 9], C + [R + 7, R + 7, 15]
    return f, (lo, hi)


# ------------------------------------------------------------------ build

def bounds():
    return np.array([-BASE[0] - 3, -BASE[1] - 3, -0.6]), np.array([BASE[0] + 3, BASE[1] + 3, HMAX + 6])


def build(voxel, stems=(), quiet=False, region=None):
    say = (lambda *a: None) if quiet else print
    t0 = time.time()
    lo, hi = bounds() if region is None else (np.asarray(region[0]), np.asarray(region[1]))
    xs = np.arange(lo[0], hi[0], voxel, dtype=np.float32)
    ys = np.arange(lo[1], hi[1], voxel, dtype=np.float32)
    zs = np.arange(lo[2], hi[2], voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    add(vol, xs, ys, zs, (-BASE[0] - 3, -BASE[1] - 3, -1), (BASE[0] + 3, BASE[1] + 3, RIM_H + 4), base_sdf, k=0, chunk=8)
    say(f"  base {time.time() - t0:.0f}s")

    # the wave, slice by slice, blended into the base
    holes = lace_holes()
    kb = int(np.searchsorted(zs, 6.0))
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    for i, x in enumerate(xs):
        if x < X0 - 14 or x > X1 + 1:
            continue
        d = wave_slice(float(x), Y, Z).astype(np.float32)
        for hx, hz, rx, rz in holes:
            if abs(x - hx) < rx:
                cyl = (np.hypot((x - hx) / rx, (Z - hz) / rz) - 1) * rz
                cyl = np.maximum(cyl, np.abs(Y - bend(hx) + 26.5 * (0.42 + 0.58 * height(hx) / HMAX)) - 6.5)
                d = np.maximum(d, -cyl)
        vol[i] = smin(vol[i], d, 2.0)
        # hollow: the wave's inside, WALL in from its skin, carried straight
        # down through the base so it opens on the plate, vented to the back
        # rim by tunnels with 45-degree pitched roofs
        cav = d + WALL
        cav[:, Z[0] < 6.0] = cav[:, kb:kb + 1]
        for xt in TUNNELS:
            if abs(x - xt) < 4:
                cav = np.minimum(cav, np.maximum(Z + abs(x - xt) - 2.6, -(Y - bend(x) + 4.0)))
        vol[i] = np.maximum(vol[i], -cav)
    say(f"  wave {time.time() - t0:.0f}s")

    for f, (a, b) in (cowrie_sdf(), cone_sdf()):
        add(vol, xs, ys, zs, a, b, f, k=0.6, chunk=16)
    say(f"  shells {time.time() - t0:.0f}s")

    # foam: pits cut along the crest, the lip, the roll and the swash line
    idx = np.argwhere(np.abs(vol) < 1.0)
    idx = idx[np.all((idx > 0) & (idx < np.array(vol.shape) - 1), axis=1)]
    for c0 in range(0, len(idx), 4_000_000):
        ii = idx[c0:c0 + 4_000_000]
        a_, b_, c_ = ii[:, 0], ii[:, 1], ii[:, 2]
        px, py, pz = xs[a_], ys[b_], zs[c_]
        wz = foam_zone(px, py, pz)
        m = wz > 0.25
        if not m.any():
            continue
        a_, b_, c_, px, py, pz = a_[m], b_[m], c_[m], px[m], py[m], pz[m]
        g = np.stack([vol[a_ + 1, b_, c_] - vol[a_ - 1, b_, c_], vol[a_, b_ + 1, c_] - vol[a_, b_ - 1, c_],
                      vol[a_, b_, c_ + 1] - vol[a_, b_, c_ - 1]], -1)
        g /= np.linalg.norm(g, axis=1)[:, None] + 1e-9
        vol[a_, b_, c_] = foam_pits(px, py, pz, vol[a_, b_, c_], g)
    say(f"  foam {time.time() - t0:.0f}s")

    f, (a, b) = plumeria_sdf()
    add(vol, xs, ys, zs, a, b, f, k=0.8, chunk=8)
    say(f"  plumeria {time.time() - t0:.0f}s")
    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])
    fill_voids(vol)
    drop_specks(vol, voxel, say)
    n = trim_crumbs(vol, voxel, zs)
    say(f"  sealed voids filled, specks dropped, {n} crumbs trimmed {time.time() - t0:.0f}s")
    return vol, (xs[0], ys[0], zs[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.4, help="mesh resolution, mm")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    t0 = time.time()
    vol, origin = build(args.voxel)
    mesh = mesh_volume(vol, args.voxel, origin)
    if not args.preview:
        mesh = decimate(mesh, 0.006)
    print(f"nalu: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, {time.time() - t0:.0f}s")
    out = os.path.join(args.outdir, "nalu.stl")
    mesh.export(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
