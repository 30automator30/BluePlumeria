#!/usr/bin/env python3
"""Lattice plumeria sculpture (resin / MSLA, Saturn 4 Ultra), 160 mm tall.

Golden-ratio design, after the concept sheet:

  main bloom  radius H/phi^3. Five pointed plumeria petals shaped like
              flames (swept, dipping, flicking up, twisting toward the tip),
              each a soft rim around a flowing vein lattice that holds a
              nested framed petal 1/phi its size, which holds a 1/phi^2
              solid petal. Inside sits a second whorl 1/phi the size,
              swirling the other way, then a 1/phi^2 centre whorl and a
              bead. Fan veins follow the petal's flow; cross veins are
              spaced by phi, so the cells shrink toward the tip.
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

def soft_band(d2, dz, half_w, half_t):
    """Rounded cross-section: a 2D band (d2) swept to thickness (dz), with
    its corners rounded so rims and veins read as smooth, moulded strokes."""
    r = np.minimum(half_w, half_t) * 0.95
    a, b = d2 + r, dz + r
    return (np.sqrt(np.maximum(a, 0) ** 2 + np.maximum(b, 0) ** 2)
            + np.minimum(np.maximum(a, b), 0) - r)


class FlowPetal:
    """Pointed plumeria petal shaped like a flame: rim + flowing vein
    lattice + nested framed copies.

    Petal frame: u from the claw (u=0) to the tip (u=L), v across, w off the
    face. The centreline sweeps sideways (scimitar), the face rises, dips
    and flicks up at the tip, rolls more on one edge than the other, and
    twists harder toward the tip. Rims swell at the claw and taper to the
    tip. The lattice is two families of veins: fan veins running with the
    petal's flow, crossed by curving cross-veins whose spacing shrinks by
    phi toward the tip.
    """

    def __init__(self, base, theta, elev, length, half_w, cup, curl, twist,
                 sweep=0.18, roll=0.35, nest=2, scale_t=1.0, seed=0):
        rng = np.random.default_rng(seed)
        jit = lambda a: 1 + a * rng.uniform(-1, 1)            # noqa: E731
        self.base = np.asarray(base, float)
        elev = elev + np.radians(4) * rng.uniform(-1, 1)
        d = np.array([np.cos(theta) * np.cos(elev), np.sin(theta) * np.cos(elev),
                      np.sin(elev)])
        s = np.array([-np.sin(theta), np.cos(theta), 0.0])
        self.d, self.s, self.n = d, s, np.cross(d, s)
        self.L, self.W = length * jit(0.05), half_w * jit(0.05)
        self.cup, self.curl, self.twist = cup, curl * jit(0.15), twist * jit(0.15)
        self.sweep, self.roll = sweep * jit(0.2), roll
        self.nest, self.k = nest, scale_t

    def width(self, t):
        return self.W * t ** 0.55 * (1 - t) ** 0.95 / 0.338

    def centre(self, t):
        """Sideways sweep of the centreline (scimitar curve)."""
        return self.sweep * self.L * t ** 2

    def outline(self, u, v, f=1.0, du=0.0):
        uu = (u - du) / f
        t = np.clip(uu / self.L, 0, 1)
        vv = (v - self.centre(np.clip(u / self.L, 0, 1))) / f
        half = self.width(t) * np.where(vv > 0, 1.06, 0.94) + 1e-3
        return f * np.maximum(np.abs(vv) - half, np.maximum(-uu, uu - self.L))

    def veins(self, u, v, f, du):
        """Distance to the flowing vein lattice of one (scaled) petal."""
        L = self.L * f
        t = np.clip((u - du) / L, 1e-3, 1)
        half = np.maximum(self.width(t) * f, 1e-3)
        sv = (v - self.centre(np.clip(u / self.L, 0, 1))) / half   # -1..1 across
        sv = sv + 0.07 * np.sin(2 * np.pi * 2.5 * t)               # a gentle wave
        # fan veins: 5 lines across the petal (a Fibonacci count)
        fan = np.abs(((sv + 1) * 2.5) % 1.0 - 0.5) / 2.5 * half
        # cross veins: arcs bowed toward the tip, spaced by phi
        bow = t + 0.16 * sv ** 2
        k = np.log(np.clip(1 - bow, 1e-4, 1)) / np.log(1 / PHI)    # 1 - t = phi^-k
        cross = np.abs(k - np.round(k)) * L * np.clip(1 - bow, 0.05, 1) * 0.48
        return np.minimum(fan, cross)

    def mid(self, u, v):
        t = np.clip(u / self.L, 0, 1)
        vv = (v - self.centre(t)) / self.W
        return (self.L * self.curl * (2.2 * t ** 3 - 1.0 * t ** 2)    # dip, then flick
                + self.cup * self.W * vv ** 2 * t
                + self.roll * self.W * vv ** 3 * t                    # one edge rolls
                + self.twist * self.W * vv * t ** 2 * 2.0)            # twist grows

    def sdf(self, x, y, z):
        q = np.stack([x - self.base[0], y - self.base[1], z - self.base[2]])
        u = np.tensordot(self.d, q, 1)
        v = np.tensordot(self.s, q, 1)
        w = np.tensordot(self.n, q, 1)
        h = self.mid(u, v)
        k = self.k
        t = np.clip(u / self.L, 0, 1)
        out = np.full(u.shape, 1e9, np.float32)
        for lvl in range(self.nest + 1):
            f = PHI ** -lvl
            du = self.L * 0.10 * lvl
            o = self.outline(u, v, f, du)
            lift = 0.55 * lvl * k
            dz = lambda thick: np.abs(w - h - lift) - thick / 2      # noqa: E731
            if lvl == self.nest:
                # innermost: a soft solid petal with a raised centre vein
                core_t = RIM_T * k * (1.15 - 0.5 * t)
                out = smin(out, soft_band(o, dz(core_t), 5.0, core_t / 2), 0.4)
                vein = np.maximum(np.abs(v - self.centre(t)) - 0.4 * k, o + 0.9 * k)
                out = smin(out, soft_band(vein, np.abs(w - h - lift - 0.5 * k) - 0.6 * k,
                                          0.4 * k, 0.6 * k), 0.3)
                break
            rim_w = RIM_W * k * (1.35 - 0.6 * t)         # swells at the claw
            rim_t = RIM_T * k * (1.2 - 0.45 * t)
            rim = np.maximum(o, -(o + rim_w))
            out = smin(out, soft_band(rim, dz(rim_t), rim_w / 2, rim_t / 2), 0.5)
            inner = o + rim_w * 0.6
            if lvl < self.nest:
                nxt = self.outline(u, v, PHI ** -(lvl + 1), self.L * 0.10 * (lvl + 1))
                inner = np.maximum(inner, -(nxt - GAP * k))
            sw = STRUT_W * k * (1.2 - 0.4 * t)
            strut = np.maximum(self.veins(u, v, f, du) - sw / 2, inner)
            out = smin(out, soft_band(strut, np.abs(w - h - lift + 0.25 * k) - STRUT_T * k / 2,
                                      sw / 2, STRUT_T * k / 2), 0.45)
            if lvl < self.nest:
                # the nested copy grows out of its frame at the claw
                stalk = np.maximum(np.abs(v - self.centre(t)) - 1.0 * k,
                                   np.maximum(o, u - (self.L * 0.10 * (lvl + 1) + 3.0 * k)))
                out = smin(out, soft_band(stalk, dz(rim_t), 1.0 * k, rim_t / 2), 0.6)
        return out

    def bbox(self, pad=3.0):
        pts = []
        for u in np.linspace(0, self.L, 24):
            t = u / self.L
            for vv in np.linspace(-1.1, 1.1, 7):
                v = vv * self.width(t) + self.centre(t)
                pts.append(self.base + u * self.d + v * self.s + self.mid(u, v) * self.n)
        pts = np.array(pts)
        return pts.min(0) - pad, pts.max(0) + pad


def bloom_petals(R, scale_t=1.0, centre=(0, 0, 0), seed=1):
    """Main bloom: 5 flame petals, a 1/phi whorl inside, a 1/phi^2 centre.
    Whorls alternate their sweep so the bloom swirls."""
    c = np.asarray(centre, float)
    petals = []
    whorls = [  # length, elevation, nest depth, cup, curl, twist, sweep
        (R, np.radians(16), 2, 0.30, 0.16, 0.20, 0.20),
        (R / PHI, np.radians(40), 1, 0.40, 0.20, 0.26, -0.16),
        (R / PHI ** 2, np.radians(64), 0, 0.50, 0.16, 0.30, 0.14),
    ]
    for kw, (L, elev, nest, cup, curl, twist, sweep) in enumerate(whorls):
        W = L * 0.36
        r0 = 3.0 * PHI ** -kw + 0.8
        rot = 0.3 + kw * np.pi / 5
        for i in range(5):
            th = rot + 2 * np.pi * i / 5
            off = 0.30 * W
            base = c + np.array([r0 * np.cos(th) - off * np.sin(th),
                                 r0 * np.sin(th) + off * np.cos(th), 1.4 * kw])
            petals.append(FlowPetal(base, th, elev, L, W, cup, curl, twist,
                                    sweep=sweep, nest=nest, scale_t=scale_t,
                                    seed=seed * 100 + kw * 10 + i))
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
    ext = BLOOM_R + 9
    xs = np.arange(-ext, ext, voxel, dtype=np.float32)
    ys = xs.copy()
    zs = np.arange(-12.0, 40.0, voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    for p in bloom_petals(BLOOM_R):
        lo, hi = p.bbox()
        add(vol, xs, ys, zs, lo, hi, p.sdf, k=0.6)

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
