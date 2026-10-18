#!/usr/bin/env python3
"""Lattice plumeria sculpture (resin / MSLA, Saturn 4 Ultra), 160 mm tall.

Golden-ratio design, after the concept sheet:

  main bloom  radius H/phi^3. Five pointed, curling plumeria petals, each a
              solid rim around an openwork lattice that holds a nested
              framed petal 1/phi its size, which holds a 1/phi^2 solid petal.
              Inside sits a second whorl 1/phi the size, then a 1/phi^2
              centre whorl and a bead: petals within petals at two scales.
              The lattice cells are laid out on a golden-angle (Vogel)
              spiral, so even the lace follows phi.
  body        (printed as one part) broad twisting ribbons rising from the
              base as golden spirals, pinching at a neck at H/phi^2, then
              fanning out to a second bloom (1/phi of the main) and to small
              flowers that spiral up at the golden angle. Thin 8 + 5 strands
              inside the ribbons form the lattice core.

The main bloom fits the body on a keyed peg. Every part is a signed
distance field, meshed with marching cubes into a watertight STL.

Usage
  python3 lattice_plumeria.py --part bloom --voxel 0.2   # quick look
  python3 lattice_plumeria.py                            # both parts
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "plumeria-ring"))
from fractal_plumeria import (GOLDEN_ANGLE, PHI, capsule_chain, keyed_cylinder,  # noqa: E402
                              mesh_volume)
from plumeria_ring import decimate, smin  # noqa: E402

H = 160.0                       # overall height
BLOOM_R = H / PHI ** 3          # 37.8: main petal length
RIM_W, RIM_T = 1.7, 1.9         # petal rim: width, thickness
STRUT_W, STRUT_T = 0.9, 1.25    # lattice struts: width, thickness
GAP = 1.1                       # void between a frame and what it holds
PEG_R, PEG_H, FIT = 2.6, 7.0, 0.15


# ------------------------------------------------------------ petal maths

def vogel_seeds(n):
    """n points in the unit disk on a golden-angle (sunflower) spiral."""
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n)
    a = i * GOLDEN_ANGLE
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


class LatticePetal:
    """Pointed plumeria petal: rim + Vogel lattice + nested framed copies.

    The petal lives in its own frame: u runs from the claw (u=0) to the tip
    (u=L), v across it, w off its face. The face is curved (cup across,
    curl along, pinwheel twist), and the nested copies sit progressively
    higher on it.
    """

    def __init__(self, base, theta, elev, length, half_w, cup, curl, twist,
                 nest=2, solid_core=True, scale_t=1.0):
        self.base = np.asarray(base, float)
        d = np.array([np.cos(theta) * np.cos(elev), np.sin(theta) * np.cos(elev),
                      np.sin(elev)])
        s = np.array([-np.sin(theta), np.cos(theta), 0.0])
        self.d, self.s, self.n = d, s, np.cross(d, s)
        self.L, self.W = length, half_w
        self.cup, self.curl, self.twist = cup, curl, twist
        self.nest, self.solid_core, self.k = nest, solid_core, scale_t
        # lattice seeds in a unit disk, mapped onto the petal per level
        self.seeds = vogel_seeds(max(8, int(length * half_w / 9.0)))

    # 2D outline: narrow claw, broad belly, pointed tip; f scales it about
    # a claw point shifted du along the petal
    def width(self, t):
        return self.W * t ** 0.55 * (1 - t) ** 0.95 / 0.338

    def outline(self, u, v, f=1.0, du=0.0):
        uu, vv = (u - du) / f, v / f
        t = np.clip(uu / self.L, 0, 1)
        half = self.width(t) * np.where(vv > 0, 1.06, 0.94) + 1e-3
        return f * np.maximum(np.abs(vv) - half, np.maximum(-uu, uu - self.L))

    def lattice(self, u, v, f, du):
        """Distance to the Voronoi edges of the petal's seed spiral."""
        L, W = self.L * f, self.W * f
        sx = du + L * (0.52 + 0.46 * self.seeds[:, 0])
        sy = W * 0.95 * self.seeds[:, 1]
        d1 = np.full(u.shape, 1e9, np.float32)
        d2 = np.full(u.shape, 1e9, np.float32)
        for x0, y0 in zip(sx, sy):
            dd = np.hypot(u - x0, v - y0)
            d2 = np.where(dd < d1, d1, np.minimum(d2, dd))
            d1 = np.minimum(d1, dd)
        return (d2 - d1) / 2

    def mid(self, u, v):
        t = np.clip(u / self.L, 0, 1)
        return (self.cup * self.W * (v / self.W) ** 2 * t
                + self.curl * self.L * t ** 3 * 1.4              # tip flicks up
                + self.twist * t * v)

    def sdf(self, x, y, z):
        q = np.stack([x - self.base[0], y - self.base[1], z - self.base[2]])
        u = np.tensordot(self.d, q, 1)
        v = np.tensordot(self.s, q, 1)
        w = np.tensordot(self.n, q, 1)
        h = self.mid(u, v)
        k = self.k

        def slab(d2, lift, thick):
            return np.maximum(d2, np.abs(w - h - lift) - thick / 2)

        out = np.full(u.shape, 1e9, np.float32)
        hole = None
        for lvl in range(self.nest + 1):
            f = PHI ** -lvl
            du = self.L * 0.10 * lvl
            o = self.outline(u, v, f, du)
            lift = 0.55 * lvl * k
            if lvl == self.nest and self.solid_core:
                # innermost: a solid petal with a raised centre vein
                vein = np.maximum(np.abs(v) - 0.35 * k, o + 0.8 * k)
                out = np.minimum(out, slab(o, lift, RIM_T * k))
                out = np.minimum(out, slab(vein, lift + 0.45 * k, RIM_T * k))
                break
            rim = np.maximum(o, -(o + RIM_W * k))
            out = np.minimum(out, slab(rim, lift, RIM_T * k))
            inner = o + RIM_W * k                          # inside the rim
            if lvl < self.nest:
                nxt = self.outline(u, v, PHI ** -(lvl + 1), self.L * 0.10 * (lvl + 1))
                inner = np.maximum(inner, -(nxt - GAP * k))  # leave room
            lat = np.maximum(inner, self.lattice(u, v, f, du) - STRUT_W * k / 2)
            # struts sit a little below the rim so the frame reads
            out = np.minimum(out, slab(np.maximum(lat, o + 0.2), lift - 0.3 * k,
                                       STRUT_T * k))
            if lvl < self.nest:
                # a stalk ties the nested copy to its frame at the claw
                stalk = np.maximum(np.abs(v) - 0.9 * k,
                                   np.maximum(o, u - (self.L * 0.10 * (lvl + 1)
                                                      + 2.5 * k)))
                out = np.minimum(out, slab(stalk, lift, RIM_T * k))
        return out

    def bbox(self, pad=2.5):
        pts = []
        for u in np.linspace(0, self.L, 20):
            t = u / self.L
            for v in np.linspace(-1.1, 1.1, 7) * self.width(t):
                pts.append(self.base + u * self.d + v * self.s + self.mid(u, v) * self.n)
        pts = np.array(pts)
        return pts.min(0) - pad, pts.max(0) + pad


def bloom_petals(R, scale_t=1.0, centre=(0, 0, 0)):
    """Main bloom: 5 framed petals, a 1/phi whorl inside, a 1/phi^2 centre."""
    c = np.asarray(centre, float)
    petals = []
    whorls = [  # length, elevation, nest depth, cup, curl, twist
        (R, np.radians(12), 2, 0.30, 0.10, 0.22),
        (R / PHI, np.radians(36), 1, 0.38, 0.14, 0.26),
        (R / PHI ** 2, np.radians(62), 0, 0.45, 0.10, 0.30),
    ]
    for k, (L, elev, nest, cup, curl, twist) in enumerate(whorls):
        W = L * 0.36
        r0 = 3.0 * PHI ** -k + 0.8
        rot = 0.3 + k * np.pi / 5                      # alternate the whorls
        for i in range(5):
            th = rot + 2 * np.pi * i / 5
            off = 0.35 * W
            base = c + np.array([r0 * np.cos(th) - off * np.sin(th),
                                 r0 * np.sin(th) + off * np.cos(th), 1.4 * k])
            petals.append(LatticePetal(base, th, elev, L, W, cup, curl, twist,
                                       nest=nest, scale_t=scale_t))
    return petals


# ---------------------------------------------------------------- volumes

def sub(xs, ys, zs, lo, hi):
    i0, i1 = np.searchsorted(xs, lo[0]), np.searchsorted(xs, hi[0])
    j0, j1 = np.searchsorted(ys, lo[1]), np.searchsorted(ys, hi[1])
    k0, k1 = np.searchsorted(zs, lo[2]), np.searchsorted(zs, hi[2])
    return slice(i0, i1), slice(j0, j1), slice(k0, k1)


def add(vol, xs, ys, zs, lo, hi, fn, k=0.3, chunk=24):
    """smin fn() into vol inside a box, a few x-slabs at a time."""
    sl = sub(xs, ys, zs, lo, hi)
    for i in range(sl[0].start, sl[0].stop, chunk):
        sx = slice(i, min(i + chunk, sl[0].stop))
        x, y, z = np.meshgrid(xs[sx], ys[sl[1]], zs[sl[2]], indexing="ij")
        if x.size == 0:
            continue
        d = np.minimum(fn(x, y, z), 3.0).astype(np.float32)
        cur = vol[sx, sl[1], sl[2]]
        vol[sx, sl[1], sl[2]] = smin(cur, d, k) if k else np.minimum(cur, d)


def build_bloom(voxel):
    ext = BLOOM_R + 5
    xs = np.arange(-ext, ext, voxel, dtype=np.float32)
    ys = xs.copy()
    zs = np.arange(-12.0, 32.0, voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    for p in bloom_petals(BLOOM_R):
        lo, hi = p.bbox()
        add(vol, xs, ys, zs, lo, hi, p.sdf, k=0.25)

    # bead centre, calyx and keyed socket for the body's peg
    def bead(x, y, z):
        return np.sqrt(x ** 2 + y ** 2 + ((z - 5.5) / 1.2) ** 2) - 2.6
    add(vol, xs, ys, zs, (-5, -5, 1), (5, 5, 10), bead, k=0.6)

    def calyx(x, y, z):
        c = np.sqrt(x ** 2 + y ** 2 + ((z + 3.0) / 1.5) ** 2) - 6.8
        return np.maximum(c, -z - 10.5)
    add(vol, xs, ys, zs, (-9, -9, -12), (9, 9, 6), calyx, k=1.2)
    sl = sub(xs, ys, zs, (-5, -5, -12), (5, 5, 0))
    x, y, z = np.meshgrid(xs[sl[0]], ys[sl[1]], zs[sl[2]], indexing="ij")
    socket = keyed_cylinder(x, y, z, PEG_R + FIT, -11.0, -10.5 + PEG_H + 0.4)
    vol[sl] = np.maximum(vol[sl], -socket)
    return vol, (xs[0], ys[0], zs[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["bloom", "body", "both"], default="both")
    ap.add_argument("--voxel", type=float, default=0.12, help="mesh resolution, mm")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    for part in (["bloom", "body"] if args.part == "both" else [args.part]):
        t0 = time.time()
        vol, origin = build_bloom(args.voxel) if part == "bloom" else build_body(args.voxel)
        print(f"{part}: grid {vol.shape}, sdf {time.time() - t0:.0f}s")
        mesh = decimate(mesh_volume(vol, args.voxel, origin), 0.006)
        print(f"{part}: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
              f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, "
              f"{time.time() - t0:.0f}s")
        out = os.path.join(args.outdir, f"lattice_plumeria_{part}.stl")
        mesh.export(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
