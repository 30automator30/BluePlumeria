#!/usr/bin/env python3
"""Fractal spiral plumeria desk sculpture (resin / MSLA, Saturn 4 Ultra).

Two parts that join with a keyed peg:

  bloom  A ~65 mm plumeria: five broad pinwheel petals cradling a
         golden-angle spiral of 29 smaller plumeria petals that shrink
         toward the centre and rise from open, to cupped, to a curled bud:
         petals within petals. Every petal keeps the pinwheel twist.
  stem   Earth -> vortex -> explosion. Thirteen root spirals sweep across
         an open, flat base to a hub; from it 8 + 5 counter-wound strands
         climb as a vortex that narrows and twists faster as it rises; at
         the neck 13 strands burst outward under the bloom and curl into
         fractal tendrils (golden spirals that sprout smaller
         counter-curling spirals, three levels deep).

Each part is a signed distance field meshed with marching cubes, then
simplified to a watertight STL. Units are mm; +Z is up as displayed.

Usage
  python3 fractal_plumeria.py              # both parts
  python3 fractal_plumeria.py --part bloom --voxel 0.15   # quick look
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "plumeria-ring"))
from plumeria_ring import decimate, smin  # noqa: E402

PHI = (1 + 5 ** 0.5) / 2
GOLDEN_ANGLE = np.pi * (3 - 5 ** 0.5)          # 137.5 degrees

BLOOM_R = 32.0          # outer petal length (bloom ~65 mm across)
PEG_R, PEG_H = 2.4, 6.0  # joint peg (on the stem) / socket (in the bloom)
FIT = 0.15              # socket clearance per side
MIN_T = 0.9             # thinnest petal edge, mm


# ----------------------------------------------------------------- shapes

def frame(theta, elev):
    """Petal axes: d runs base->tip, s across, n off the petal's face."""
    d = np.array([np.cos(theta) * np.cos(elev), np.sin(theta) * np.cos(elev), np.sin(elev)])
    s = np.array([-np.sin(theta), np.cos(theta), 0.0])
    n = np.cross(d, s)
    return d, s, n


class Petal:
    """A plumeria petal: obovate outline, pinwheel twist, cup and curl."""

    def __init__(self, base, theta, elev, length, half_w, cup, curl, twist,
                 t_base, t_tip):
        self.base = np.asarray(base, float)
        self.d, self.s, self.n = frame(theta, elev)
        self.L, self.W = length, half_w
        self.cup, self.curl, self.twist = cup, curl, twist
        self.tb, self.tt = t_base, t_tip

    def width(self, t):
        # narrow claw, broad rounded tip; slightly wider on the twist side
        w = t ** 0.55 * np.sqrt(np.clip(1 - t ** 3.0, 0, 1))
        return self.W * w / 0.80

    def mid(self, u, v):
        t = np.clip(u / self.L, 0, 1)
        return (self.cup * self.W * (v / self.W) ** 2 * t
                + self.curl * self.L * t ** 2
                + self.twist * t * v)

    def sdf(self, x, y, z):
        qx, qy, qz = x - self.base[0], y - self.base[1], z - self.base[2]
        u = qx * self.d[0] + qy * self.d[1] + qz * self.d[2]
        v = qx * self.s[0] + qy * self.s[1] + qz * self.s[2]
        w = qx * self.n[0] + qy * self.n[1] + qz * self.n[2]
        t = np.clip(u / self.L, 0, 1)
        d2 = self.outline(u, v, 1.0)
        h = self.mid(u, v)
        dh_du = (self.cup * self.W * (v / self.W) ** 2 / self.L
                 + 2 * self.curl * t + self.twist * v / self.L)
        dh_dv = 2 * self.cup * v / self.W * t + self.twist * t
        g = np.sqrt(1 + dh_du ** 2 + dh_dv ** 2)
        thick = self.tt + (self.tb - self.tt) * (1 - t)
        return np.maximum(d2, np.abs(w - h) / g - thick / 2)

    def outline(self, u, v, f):
        """2D distance to the petal outline scaled by f about the claw."""
        uu, vv = u / f, v / f
        t = np.clip(uu / self.L, 0, 1)
        half = self.width(t) * np.where(vv > 0, 1.08, 0.92) + 1e-3
        return f * np.maximum(np.abs(vv) - half, np.maximum(-uu, uu - self.L))

    def bbox(self, pad=1.0):
        us = np.linspace(0, self.L, 24)
        pts = []
        for u in us:
            t = u / self.L
            for v in np.linspace(-1.1, 1.1, 9) * self.width(t):
                p = self.base + u * self.d + v * self.s + self.mid(u, v) * self.n
                pts.append(p)
        pts = np.array(pts)
        return pts.min(0) - pad - self.tb, pts.max(0) + pad + self.tb


def capsule_chain(x, y, z, pts, radii):
    """Distance to a tube of varying radius through 3D points."""
    d = np.full(x.shape, 1e9, np.float32)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        L2 = max(ab @ ab, 1e-12)
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1] + (z - a[2]) * ab[2]) / L2, 0, 1)
        r = radii[i] + t * (radii[i + 1] - radii[i])
        dd = np.sqrt((x - a[0] - t * ab[0]) ** 2 + (y - a[1] - t * ab[1]) ** 2
                     + (z - a[2] - t * ab[2]) ** 2) - r
        d = np.minimum(d, dd)
    return d


def keyed_cylinder(x, y, z, r, z0, z1):
    """Vertical cylinder with a flat on its +x side (stops rotation)."""
    rr = np.sqrt(x * x + y * y)
    return np.maximum(np.maximum(rr - r, x - 0.72 * r), np.maximum(z0 - z, z - z1))


# ------------------------------------------------------------------ bloom

N_PETALS = 34            # Fibonacci: 5- and 8-armed spirals show on the face


def bloom_petals():
    """A classic five-petal plumeria pinwheel cradling a golden-angle
    spiral of plumeria petals that shrink by phi^3 toward the centre and
    rise from open, to cupped, to a curled bud: petals within petals."""
    petals = []
    # outer tier: the plumeria itself, five broad pinwheel petals
    for i in range(5):
        theta = 0.3 + 2 * np.pi * i / 5
        L, W = BLOOM_R, BLOOM_R * 0.50
        off = 0.42 * W
        base = np.array([4.2 * np.cos(theta) - off * np.sin(theta),
                         4.2 * np.sin(theta) + off * np.cos(theta), -0.6])
        petals.append(Petal(base, theta, np.radians(14), L, W, cup=0.32,
                            curl=-0.12, twist=0.30, t_base=1.7, t_tip=1.15))
    # inner spiral: the fractal heart
    n = N_PETALS - 5
    for i in range(n):
        p = i / (n - 1)                          # 0 = outer, 1 = core
        theta = i * GOLDEN_ANGLE
        L = BLOOM_R / PHI * PHI ** (-2.2 * p)
        W = L * (0.47 - 0.05 * p)
        elev = np.radians(24 + 60 * p ** 1.1)
        r0 = 2.8 * (1 - p) + 0.9
        off = 0.30 * W                           # pinwheel offset
        base = np.array([r0 * np.cos(theta) - off * np.sin(theta),
                         r0 * np.sin(theta) + off * np.cos(theta),
                         1.0 + 5.0 * p ** 0.8])
        petals.append(Petal(base, theta, elev, L, W,
                            cup=0.40 + 0.40 * p, curl=0.02 + 0.34 * p,
                            twist=0.26 + 0.08 * p,
                            t_base=1.5 - 0.35 * p, t_tip=max(MIN_T, 1.05 - 0.15 * p)))
    return petals


def bloom_sdf_into(vol, xs, ys, zs):
    """Write the bloom's SDF into `vol` on the grid (xs, ys, zs)."""
    def sub(lo, hi):
        i0, i1 = np.searchsorted(xs, lo[0]), np.searchsorted(xs, hi[0])
        j0, j1 = np.searchsorted(ys, lo[1]), np.searchsorted(ys, hi[1])
        k0, k1 = np.searchsorted(zs, lo[2]), np.searchsorted(zs, hi[2])
        return slice(i0, i1), slice(j0, j1), slice(k0, k1)

    def grid(sl):
        return np.meshgrid(xs[sl[0]], ys[sl[1]], zs[sl[2]], indexing="ij")

    for p in bloom_petals():
        sl = sub(*p.bbox())
        x, y, z = grid(sl)
        vol[sl] = smin(vol[sl], np.minimum(p.sdf(x, y, z), 3.0), 0.35)

    # calyx under the petals, with the keyed socket for the stem's peg
    sl = sub(np.array([-9, -9, -11.0]), np.array([9, 9, 6.0]))
    x, y, z = grid(sl)
    calyx = np.sqrt(x ** 2 + y ** 2 + ((z + 2.5) / 1.6) ** 2) * 1.0 - 6.5
    calyx = np.maximum(calyx, -z - 9.5)                      # flat foot
    vol[sl] = smin(vol[sl], calyx, 1.2)
    socket = keyed_cylinder(x, y, z, PEG_R + FIT, -10.0, -9.5 + PEG_H + 0.4)
    vol[sl] = np.maximum(vol[sl], -socket)

    # tiny closed bud at the very centre
    sl = sub(np.array([-4, -4, 1.0]), np.array([4, 4, 11.0]))
    x, y, z = grid(sl)
    bud = np.sqrt(x ** 2 + y ** 2 + ((z - 6.5) / 1.4) ** 2) - 2.2
    vol[sl] = smin(vol[sl], bud, 0.6)


# ------------------------------------------------------------------- stem
#
# Earth -> vortex -> explosion. Root spirals sweep inward across a flat,
# open base to a hub; from it two counter-wound sets of strands (8 and 5,
# a Fibonacci pair) climb as a vortex that narrows and twists faster as
# it rises, concentrating at a neck; above the neck 13 strands burst
# outward under the bloom and curl into fractal tendrils.

# Golden proportions (all lengths mm). The sculpture is H tall; its
# narrowest point (the neck) sits at H/phi. Below it the vortex contracts,
# above it the burst expands: each quarter turn the strands' radius scales
# by phi and their rise by phi^2, which gives a flared, trumpet outline.
H_TOTAL = 122.0
NECK_Z = H_TOTAL / PHI                   # 75.4: vortex : burst = phi : 1
BASE_R = 27.0                            # R0
NECK_R = BASE_R / PHI ** 4               # 3.9
VORTEX_QT = 4.0                          # quarter turns, base -> neck
BURST_QT = 4.5                           # quarter turns, neck -> bloom (R0*sqrt(phi))
BLOOM_H = 24.6                           # bloom height above its foot
STEM_TOP = H_TOTAL - BLOOM_H             # the bloom's foot sits here
TILT = 0.0                               # rises straight up
STRAND_MIN = 0.6                         # thinnest tendril radius


def vortex_z(k):
    """Height after k quarter turns up the vortex (rise / phi^2 each)."""
    dz0 = NECK_Z * (1 - PHI ** -2) / (1 - PHI ** (-2 * VORTEX_QT))
    return dz0 * (1 - PHI ** (-2 * k)) / (1 - PHI ** -2)


def burst_z(k):
    """Height after k quarter turns up from the neck (rise * phi^2 each)."""
    top = H_TOTAL - NECK_Z
    a = top * (PHI ** 2 - 1) / (PHI ** (2 * BURST_QT) - 1)
    return NECK_Z + a * (PHI ** (2 * k) - 1) / (PHI ** 2 - 1)


def log_spiral(origin, start_dir, up, r0, turns, sense, n=90):
    """Golden spiral curling in the plane normal to `up`; starts at
    `origin` heading along `start_dir`."""
    b = np.log(PHI) / (np.pi / 2)            # grows by phi each quarter turn
    th = np.linspace(0, 2 * np.pi * turns, n)
    rad = r0 * np.exp(-b * th)               # curls inward
    e1 = start_dir / np.linalg.norm(start_dir)
    e2 = np.cross(up, e1)
    e2 /= np.linalg.norm(e2)
    centre = origin + r0 * sense * e2
    return (centre[None] - rad[:, None] * sense * (np.cos(th)[:, None] * e2[None]
                                                - np.sin(th)[:, None] * sense * e1[None]))


def tendril(chains, origin, direction, normal, size, sense, r0, depth):
    """A golden-spiral tendril that sprouts two smaller, counter-curling
    tendrils of its own, `depth` levels deep."""
    pts = log_spiral(origin, direction, normal, size, 1.15, sense, n=int(40 + 8 * size))
    rr = np.linspace(r0, max(STRAND_MIN, r0 * 0.5), len(pts))
    chains.append((pts, rr))
    if depth > 1:
        for f, flip in ((0.22, 1), (0.48, -1)):
            i = int(f * (len(pts) - 1))
            tan = pts[i + 1] - pts[i]
            tan /= np.linalg.norm(tan)
            side = flip * np.cross(normal, tan)
            tendril(chains, pts[i], side + 0.3 * tan, tan, size / PHI ** 2,
                    -sense, max(STRAND_MIN, rr[i] * 0.8), depth - 1)


def helix(r_of_s, th_of_s, z0, z1, n=160):
    s = np.linspace(0, 1, n)
    r, th = r_of_s(s), th_of_s(s)
    return np.column_stack([r * np.cos(th), r * np.sin(th), z0 + (z1 - z0) * s])


def tip_point():
    return np.array([0.0, 0.0, STEM_TOP])


def stem_parts():
    """All tube chains (points, radii) making up the stem part."""
    chains = []
    # roots: 13 golden spirals sweeping inward across the ground, from the
    # outer ring to the hub (radius / phi each quarter turn)
    kr = np.log(BASE_R / 7.0) / np.log(PHI)
    for i in range(13):
        k = np.linspace(0, kr, 60)
        r = (BASE_R - 1.0) * PHI ** -k
        th = 2 * np.pi * i / 13 + k * np.pi / 2
        pts = np.column_stack([r * np.cos(th), r * np.sin(th), np.full_like(k, 1.25)])
        chains.append((pts, np.full(len(k), 1.25)))
    # the 8 + 5 strands: up the vortex, through the neck, out into the burst
    burst_ends = []
    for count, sense, phase in ((8, 1, 0.0), (5, -1, np.pi / 5)):
        for i in range(count):
            a0 = phase + 2 * np.pi * i / count
            kv = np.linspace(0, VORTEX_QT, 140)
            kb = np.linspace(0, BURST_QT * 0.86, 80)[1:]    # stop under the petals
            r = np.concatenate([(BASE_R - 1.0) * PHI ** -kv, NECK_R * PHI ** kb])
            z = np.concatenate([1.2 + (vortex_z(kv) - 1.2 * kv / VORTEX_QT), burst_z(kb)])
            kk = np.concatenate([kv, VORTEX_QT + kb])
            th = a0 + sense * kk * np.pi / 2
            pts = np.column_stack([r * np.cos(th), r * np.sin(th), z])
            # thick at the root, slimmest at the neck, filling out again
            rad = np.concatenate([np.linspace(1.3, 0.8, len(kv)),
                                  np.linspace(0.8, 1.05, len(kb))])
            chains.append((pts, rad))
            burst_ends.append((pts, rad, sense))
    # core: solid column up the middle, carrying the joint peg
    z = np.linspace(1.0, STEM_TOP - 3.2, 40)        # the collar caps it
    rad = np.where(z < NECK_Z, NECK_R - 0.6 + 1.6 * (1 - z / NECK_Z) ** 2, NECK_R - 1.1)
    chains.append((np.column_stack([np.zeros_like(z), np.zeros_like(z), z]), rad))
    # every burst strand ends in a fractal tendril (3 or 2 levels)
    for j, (pts, rad, sense) in enumerate(burst_ends):
        out = pts[-1] - pts[-2]
        out /= np.linalg.norm(out)
        radial = np.array([pts[-1][0], pts[-1][1], 0.0])
        radial /= np.linalg.norm(radial)
        normal = np.cross(radial, [0, 0, 1.0])
        # curl outward and down, like falling sparks, clear of the petals
        tendril(chains, pts[-1], radial + 0.5 * np.array([0, 0, 1.0]), normal,
                4.0 + 1.3 * (j % 3), -1, rad[-1], 3 if j % 2 == 0 else 2)
    # and a few flung off the vortex on the way up, at golden heights
    for j, kq in enumerate([1.0, 1.0 + 1 / PHI, 2.0 + 1 / PHI]):
        pts, rad = chains[13 + 3 * j]
        i = int(kq / VORTEX_QT * 139)
        p = pts[i]
        radial = np.array([p[0], p[1], 0.0])
        radial /= np.linalg.norm(radial)
        tendril(chains, p, radial + np.array([0, 0, 0.5]), np.cross(radial, [0, 0, 1.0]),
                9.0 / PHI ** j, 1 if j % 2 else -1, rad[i] * 0.9, 3)
    return chains


def _sub(xs, ys, zs, lo, hi):
    i0, i1 = np.searchsorted(xs, lo[0]), np.searchsorted(xs, hi[0])
    j0, j1 = np.searchsorted(ys, lo[1]), np.searchsorted(ys, hi[1])
    k0, k1 = np.searchsorted(zs, lo[2]), np.searchsorted(zs, hi[2])
    sl = slice(i0, i1), slice(j0, j1), slice(k0, k1)
    return sl, np.meshgrid(xs[sl[0]], ys[sl[1]], zs[sl[2]], indexing="ij")


def tip_frame(x, y, z, tip):
    """Coordinates along the tilted stem tip: lz runs up the tip axis."""
    c, s_ = np.cos(TILT), np.sin(TILT)
    qx, qy, qz = x - tip[0], y - tip[1], z - tip[2]
    return qx, qy * c + qz * s_, -qy * s_ + qz * c


def stem_sdf_into(vol, xs, ys, zs):
    for pts, rr in stem_parts():
        # evaluate long chains in short runs, each inside its own box
        for k in range(0, len(pts) - 1, 10):
            seg, rad = pts[k:k + 11], rr[k:k + 11]
            pad = rad.max() + 0.8
            sl, (x, y, z) = _sub(xs, ys, zs, seg.min(0) - pad, seg.max(0) + pad)
            if x.size == 0:
                continue
            d = np.minimum(capsule_chain(x, y, z, seg, rad), 3.0)
            vol[sl] = smin(vol[sl], d, 0.35)

    # solid outer ring and hub tie the roots into a flat, stable foot
    sl, (x, y, z) = _sub(xs, ys, zs, np.array([-BASE_R - 1, -BASE_R - 1, -1]),
                         np.array([BASE_R + 1, BASE_R + 1, 6]))
    rr = np.sqrt(x ** 2 + y ** 2)
    ring = np.sqrt(np.maximum(np.abs(rr - (BASE_R - 1.6)) - 0.4, 0) ** 2
                   + (z - 1.3) ** 2) - 1.3
    hub = np.maximum(rr - 7.2, np.abs(z - 1.6) - 1.6)
    foot = smin(ring, hub, 0.5)
    vol[sl] = smin(vol[sl], np.minimum(foot, 3.0), 0.6)
    vol[sl] = np.maximum(vol[sl], -z)                    # flat underside

    # sepal collar under the bloom, and the keyed peg, along the tilted tip
    tip = tip_point()
    sl, (x, y, z) = _sub(xs, ys, zs, tip - 9, tip + 9)
    lx, ly, lz = tip_frame(x, y, z, tip)
    rr = np.sqrt(lx ** 2 + ly ** 2)
    collar = np.maximum(rr - 4.6 + 1.2 * np.clip(-lz / 3.0, 0, 1),
                        np.maximum(lz, -4.0 - lz))
    vol[sl] = smin(vol[sl], collar, 0.8)
    vol[sl] = np.minimum(vol[sl], keyed_cylinder(lx, ly, lz, PEG_R, -0.5, PEG_H))


# ---------------------------------------------------------------- meshing

def mesh_volume(vol, voxel, origin):
    from skimage.measure import marching_cubes
    import trimesh
    vol = np.pad(vol, 1, constant_values=1.0)
    vol[vol == 0] = 1e-6          # exact zeros make coincident, unjoined vertices
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(voxel,) * 3)
    verts += np.asarray(origin) - voxel
    mesh = trimesh.Trimesh(verts, faces, process=False)
    parts = mesh.split(only_watertight=False)
    big = max(parts, key=lambda m: len(m.faces))
    if len(parts) > 1:
        print(f"  dropped {len(parts) - 1} loose fragment(s)")
    return big


def build(part, voxel):
    t0 = time.time()
    if part == "bloom":
        ext = BLOOM_R + 4
        xs = np.arange(-ext, ext, voxel, dtype=np.float32)
        ys = xs.copy()
        zs = np.arange(-11.0, 30.0, voxel, dtype=np.float32)
        vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
        bloom_sdf_into(vol, xs, ys, zs)
    else:
        pts = np.vstack([p for p, _ in stem_parts()])
        lo = np.minimum(pts.min(0) - 3, [-BASE_R - 1, -BASE_R - 1, -0.5])
        hi = np.maximum(pts.max(0) + 3, [BASE_R + 1, BASE_R + 1, STEM_TOP + PEG_H + 1])
        xs = np.arange(lo[0], hi[0], voxel, dtype=np.float32)
        ys = np.arange(lo[1], hi[1], voxel, dtype=np.float32)
        zs = np.arange(lo[2], hi[2], voxel, dtype=np.float32)
        vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
        stem_sdf_into(vol, xs, ys, zs)
    print(f"{part}: grid {vol.shape}, sdf {time.time() - t0:.0f}s")
    mesh = mesh_volume(vol, voxel, (xs[0], ys[0], zs[0]))
    mesh = decimate(mesh, 0.006)
    print(f"{part}: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, "
          f"{time.time() - t0:.0f}s")
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["bloom", "stem", "both"], default="both")
    ap.add_argument("--voxel", type=float, default=0.1, help="mesh resolution, mm")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    for part in (["bloom", "stem"] if args.part == "both" else [args.part]):
        mesh = build(part, args.voxel)
        out = os.path.join(args.outdir, f"fractal_plumeria_{part}.stl")
        mesh.export(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
