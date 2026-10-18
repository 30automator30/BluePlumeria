#!/usr/bin/env python3
"""Fractal spiral plumeria desk sculpture (resin / MSLA, Saturn 4 Ultra).

Two parts that join with a keyed peg:

  bloom  A ~65 mm plumeria: five broad pinwheel petals cradling a
         golden-angle spiral of 29 smaller plumeria petals that shrink
         toward the centre and rise from open, to cupped, to a curled bud:
         petals within petals. Every petal keeps the pinwheel twist.
  stem   A gently S-curved stem rising from a hollow domed base. Tendrils
         branch off it and curl into golden (logarithmic) spirals, and
         each tendril throws off a smaller counter-curling spiral: the
         same recursion as the bloom.

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

STEM_TOP = 92.0          # height of the stem tip (bloom sits on it)
TILT = np.radians(25)    # stem tip leans forward so the bloom faces you
BASE_R, BASE_H, BASE_WALL = 26.0, 11.0, 2.4


def log_spiral(origin, start_dir, up, r0, turns, sense, n=90):
    """Golden spiral curling in a tilted plane; returns points and radii."""
    b = np.log(PHI) / (np.pi / 2)            # grows by phi each quarter turn
    th = np.linspace(0, 2 * np.pi * turns, n)
    rad = r0 * np.exp(-b * th)               # curls inward
    e1 = start_dir / np.linalg.norm(start_dir)
    e2 = np.cross(up, e1)
    e2 /= np.linalg.norm(e2)
    centre = origin + r0 * sense * e2
    pts = (centre[None] - rad[:, None] * sense * (np.cos(th)[:, None] * e2[None]
                                               - np.sin(th)[:, None] * sense * e1[None]))
    return pts


def stem_parts():
    """Main stem polyline plus spiral tendrils (two levels)."""
    t = np.linspace(0, 1, 60)
    z0 = BASE_H - 2
    stem = np.column_stack([6.0 * np.sin(np.pi * t * 1.1) * (1 - t) ** 0.3,
                            2.5 * np.sin(2 * np.pi * t),
                            z0 + (STEM_TOP - z0) * t])
    stem[-8:, :2] *= np.linspace(1, 0, 8)[:, None]      # finish over centre
    # lean the top third forward (-y) so the tip leaves at angle TILT
    zs_ = z0 + (STEM_TOP - z0) * 0.65
    sb = np.clip((stem[:, 2] - zs_) / (STEM_TOP - zs_), 0, 1)
    stem[:, 1] -= np.tan(TILT) * (STEM_TOP - zs_) * sb ** 2 / 2
    stem_r = 3.3 - 1.0 * t
    tendrils = []
    for j, (h, size) in enumerate([(0.24, 20.0), (0.44, 16.0), (0.62, 12.0)]):
        i = int(h * (len(t) - 1))
        p = stem[i]
        az = 0.6 + j * GOLDEN_ANGLE
        out = np.array([np.cos(az), np.sin(az), 0.55])
        up = np.array([-np.sin(az), np.cos(az), 0.25])
        sense = 1 if j % 2 == 0 else -1
        pts = log_spiral(p, out, up / np.linalg.norm(up), size, 1.2, sense)
        rr = np.linspace(stem_r[i] * 0.55, 0.62, len(pts))
        tendrils.append((pts, rr))
        # second level: a smaller counter-curling spiral off the tendril
        q = pts[len(pts) // 4]
        tan = pts[len(pts) // 4 + 1] - q
        sub = log_spiral(q, np.cross(tan, up) + 0.4 * up, tan / np.linalg.norm(tan),
                         size / PHI ** 1.6, 1.0, -sense, n=60)
        tendrils.append((sub, np.linspace(rr[len(pts) // 4] * 0.75, 0.55, len(sub))))
    return stem, stem_r, tendrils


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
    stem, stem_r, tendrils = stem_parts()
    for pts, rr, k in [(stem, stem_r, 0.0)] + [(p, r, 0.6) for p, r in tendrils]:
        pad = rr.max() + 1.0
        sl, (x, y, z) = _sub(xs, ys, zs, pts.min(0) - pad, pts.max(0) + pad)
        d = np.minimum(capsule_chain(x, y, z, pts, rr), 3.0)
        vol[sl] = smin(vol[sl], d, k) if k else np.minimum(vol[sl], d)

    # sepal collar under the bloom, and the keyed peg, along the tilted tip
    tip = stem[-1]
    sl, (x, y, z) = _sub(xs, ys, zs, tip - 9, tip + 9)
    lx, ly, lz = tip_frame(x, y, z, tip)
    rr = np.sqrt(lx ** 2 + ly ** 2)
    collar = np.maximum(rr - 4.6 + 1.2 * np.clip(-lz / 3.0, 0, 1),
                        np.maximum(lz, -4.0 - lz))
    vol[sl] = smin(vol[sl], collar, 0.8)
    vol[sl] = np.minimum(vol[sl], keyed_cylinder(lx, ly, lz, PEG_R, -0.5, PEG_H))

    # hollow domed base, open underneath (no trapped resin), with five
    # leaf ridges spiralling up it
    sl, (x, y, z) = _sub(xs, ys, zs, np.array([-BASE_R - 1, -BASE_R - 1, -1]),
                         np.array([BASE_R + 1, BASE_R + 1, BASE_H + 2]))
    rr = np.sqrt(x ** 2 + y ** 2)
    dome = np.sqrt(rr ** 2 + (z * BASE_R / BASE_H) ** 2) * BASE_H / BASE_R - BASE_H
    dome = np.maximum(dome, -z)
    hi = BASE_H - BASE_WALL
    ri = BASE_R - BASE_WALL                     # keeps a solid rim at the foot
    inner = np.sqrt(rr ** 2 + (z * ri / hi) ** 2) * hi / ri - hi
    shell = np.maximum(dome, -inner)
    ang = np.arctan2(y, x)
    for i in range(5):
        a0 = i * 2 * np.pi / 5
        da = np.angle(np.exp(1j * (ang - a0 - 1.1 * (1 - rr / BASE_R))))
        w = 0.30 * np.clip(np.sin(np.pi * np.clip(rr / BASE_R, 0, 1)), 0, 1) ** 0.7 + 0.02
        ridge = np.maximum(np.abs(da) * rr - w * rr, dome - 0.8)
        ridge = np.maximum(np.maximum(ridge, -dome - 0.3), rr - (BASE_R - 2.5))
        shell = smin(shell, ridge, 0.4)
    vol[sl] = smin(vol[sl], np.minimum(shell, 3.0), 2.0)


# ---------------------------------------------------------------- meshing

def mesh_volume(vol, voxel, origin):
    from skimage.measure import marching_cubes
    import trimesh
    vol = np.pad(vol, 1, constant_values=1.0)
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
        stem, _, tendrils = stem_parts()
        pts = np.vstack([stem] + [p for p, _ in tendrils])
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
