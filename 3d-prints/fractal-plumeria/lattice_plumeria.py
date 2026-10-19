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

Printability: the body prints straight on the plate with no supports. Its
base is cut flat, and plan_stems() grows S-curved stems (at most ~50 degrees
from vertical, branching like a vine) up to anything that would otherwise
start in mid-air, re-checking layer by layer until nothing is left. The main
bloom's rims and veins are bold, and every vein runs into a frame, so the
lattice has no loose ends to snap when its supports come off.

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
                              mesh_volume, tendril)
from plumeria_ring import decimate, smin  # noqa: E402

H = 160.0                       # overall height
BLOOM_R = H / PHI ** 3          # 37.8: main petal length
RIM_W, RIM_T = 2.4, 2.3         # petal rim: width, thickness (bold, so it survives supports)
STRUT_W, STRUT_T = 1.25, 1.5    # lattice struts: width, thickness
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
                # veins run on into the nested frame's rim, so no vein ends loose
                nxt = self.outline(u, v, PHI ** -(lvl + 1), self.L * 0.10 * (lvl + 1))
                inner = np.maximum(inner, -(nxt + 0.5 * RIM_W * k * (1.35 - 0.6 * t)))
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


# ------------------------------------------------------------------- body
#
# Golden layout (mm): base radius R0 = BLOOM_R / sqrt(phi); five ribbons
# spiral up from it, contracting to a neck at H / phi^2 (radius R0 / phi^3;
# radius / phi and rise / phi^2 per quarter turn). Above the neck two ribbons
# braid up the main stem to the bloom's peg, one runs to the second bloom
# (BLOOM_R / phi) and two sweep out into golden curls. Eight small flowers
# (BLOOM_R / phi^3 and / phi^4) spiral up the body at the golden angle.

R0 = BLOOM_R / PHI ** 0.5                # 29.7
NECK_Z = H / PHI ** 2                    # 61.1
NECK_R = R0 / PHI ** 3                   # 7.0
TIP = np.array([-3.0, -6.0, 116.0])      # main bloom peg (collar top)
TIP_TILT = np.radians(25)                # bloom leans toward the viewer (-y)
TIP_AXIS = np.array([0.0, -np.sin(TIP_TILT), np.cos(TIP_TILT)])
BLOOM2_C = np.array([27.0, 12.0, 80.0])  # second bloom calyx
BLOOM2_AXIS = np.array([0.62, 0.25, 0.74])
RIB_T = 2.4                              # ribbon thickness
BASE_CUT = 0.6                           # flat seat: the base prints straight on the plate
STEM_R = (1.4, 0.9)                      # self-supporting stems: radius at foot, at tip


def frame_from_axis(axis):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(ref, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1), a                    # local x, y, z


def bezier(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def stem_axis(z):
    """Main stem centreline: straight to the neck, then an S up to the tip."""
    z = np.asarray(z, float)
    s = np.clip((z - NECK_Z) / (TIP[2] - NECK_Z), 0, 1)
    x = TIP[0] * (3 * s ** 2 - 2 * s ** 3) + 4.0 * np.sin(np.pi * s)
    y = TIP[1] * s ** 2
    return np.stack([x, y, z], -1)


def vortex(a0, r_start, k_max, sense, n=120):
    """Golden vortex from the base to the neck (radius / phi, rise / phi^2
    per quarter turn)."""
    k = np.linspace(0, k_max, n)
    r = r_start * PHI ** -k
    z = 1.8 + (NECK_Z - 1.8) * (1 - PHI ** (-2 * k)) / (1 - PHI ** (-2 * k_max))
    th = a0 + sense * k * np.pi / 2
    return np.column_stack([r * np.cos(th), r * np.sin(th), z]), th[-1]


def ribbon_sdf(pts, widths, ref_fn, twist=0.0):
    """Rounded flat band through pts. A frame is carried smoothly along the
    curve (each step follows the last, gently steered so the broad face
    looks along ref_fn), then the band is turned evenly about itself by
    `twist` radians over its length."""
    n = len(pts)
    tans = np.gradient(pts, axis=0)
    tans /= np.linalg.norm(tans, axis=1, keepdims=True) + 1e-9
    wds = np.zeros((n, 3))
    for i in range(n):
        target = np.cross(tans[i], ref_fn(pts[i]))
        if np.linalg.norm(target) < 1e-6:
            target = np.cross(tans[i], [1.0, 0, 0])
        target /= np.linalg.norm(target)
        if i == 0:
            wd = target
        else:
            wd = wds[i - 1] - (wds[i - 1] @ tans[i]) * tans[i]    # transport
            wd /= np.linalg.norm(wd)
            if target @ wd < 0:
                target = -target
            wd = 0.85 * wd + 0.15 * target                       # steer gently
            wd -= (wd @ tans[i]) * tans[i]
            wd /= np.linalg.norm(wd)
        wds[i] = wd
    for i in range(n):                                            # even twist
        psi = twist * i / max(n - 1, 1)
        wds[i] = wds[i] * np.cos(psi) + np.cross(tans[i], wds[i]) * np.sin(psi)
    return [(pts[i], pts[i + 1], wds[i], wds[i + 1], widths[i] / 2, widths[i + 1] / 2)
            for i in range(n - 1)]


def smooth_path(pts, widths, keep=0, passes=6, win=7):
    """Round off kinks where path pieces join. The first `keep` points (a
    ground curl) and the far end stay put."""
    p, w = pts.copy(), widths.copy()
    k = np.ones(win) / win
    lo = max(1, keep)
    for _ in range(passes):
        for c in range(3):
            sm = np.convolve(np.pad(p[:, c], win // 2, mode="edge"), k, mode="valid")
            p[lo:-1, c] = sm[lo:-1]
    return p, w


def band_eval(segs, x, y, z):
    d = np.full(x.shape, 1e9, np.float32)
    for a, b, wa, wb, h0, h1 in segs:
        ab = b - a
        L2 = max(ab @ ab, 1e-12)
        tan = ab / np.sqrt(L2)
        tr = ((x - a[0]) * ab[0] + (y - a[1]) * ab[1] + (z - a[2]) * ab[2]) / L2
        t = np.clip(tr, 0, 1)
        over = np.abs(tr - t) * np.sqrt(L2)       # distance past the segment ends
        qx, qy, qz = x - a[0] - t * ab[0], y - a[1] - t * ab[1], z - a[2] - t * ab[2]
        qx, qy, qz = qx - (tr - t) * ab[0], qy - (tr - t) * ab[1], qz - (tr - t) * ab[2]
        # width direction blended along the segment, so the band is seamless
        wx, wy, wz = (wa[0] + t * (wb[0] - wa[0]), wa[1] + t * (wb[1] - wa[1]),
                      wa[2] + t * (wb[2] - wa[2]))
        nrm = np.sqrt(wx * wx + wy * wy + wz * wz) + 1e-9
        wx, wy, wz = wx / nrm, wy / nrm, wz / nrm
        nx, ny, nz = (tan[1] * wz - tan[2] * wy, tan[2] * wx - tan[0] * wz,
                      tan[0] * wy - tan[1] * wx)
        qw = np.abs(qx * wx + qy * wy + qz * wz)
        qn = np.abs(qx * nx + qy * ny + qz * nz)
        hw = h0 + t * (h1 - h0)
        db = soft_band(qw - hw, qn - RIB_T / 2, hw, RIB_T / 2)
        db = np.where(over > 0, np.sqrt(np.maximum(db, 0) ** 2 + over ** 2), db)
        d = smin(d, db, 0.5)          # melt neighbouring pieces: no seams
    return d


def radial_ref(p):
    """Which way a ribbon's broad face looks: outward from the stem, except
    near the ground, where the roots lie flat (face up) and roll upright."""
    c = stem_axis(p[2])
    v = p - c
    v[2] = 0
    v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-3 else np.array([1.0, 0, 0])
    f = np.clip((p[2] - 2.5) / 7.0, 0, 1)
    return (1 - f) * np.array([0, 0, 1.0]) + f * v


N_RIBBONS = 8                            # Fibonacci


def ground_curl(start, tangent, size, sense):
    """A flat golden curl on the ground that runs into `start` along
    `tangent` (returned in order: curl tip -> start)."""
    t = np.array([tangent[0], tangent[1], 0.0])
    t /= np.linalg.norm(t)
    e2 = sense * np.cross([0, 0, 1.0], t)        # curl to one side
    b = np.log(PHI) / (np.pi / 2)
    ang = np.linspace(0, 2 * np.pi * 1.1, 80)
    rad = size * np.exp(-b * ang)
    centre = start - size * e2
    pts = centre + (rad[:, None] * (np.cos(ang)[:, None] * e2
                                    - np.sin(ang)[:, None] * t))
    pts[:, 2] = RIB_T / 2 + (start[2] - RIB_T / 2) * np.exp(-ang)[:]  # settle down flat
    return pts[::-1]


def golden_curl(p, tan, r0, turns=1.1, n=70):
    """Upright golden curl continuing from p along tan (curls upward)."""
    tan = tan / np.linalg.norm(tan)
    normal = np.cross(tan, [0, 0, 1.0])
    if np.linalg.norm(normal) < 1e-6:
        normal = np.array([1.0, 0, 0])
    normal /= np.linalg.norm(normal)
    e2 = np.cross(normal, tan)
    b = np.log(PHI) / (np.pi / 2)
    ang = np.linspace(0, 2 * np.pi * turns, n)[1:]
    centre = p + r0 * e2
    pts = centre - (r0 * np.exp(-b * ang))[:, None] * (np.cos(ang)[:, None] * e2
                                                       - np.sin(ang)[:, None] * tan)
    return pts, centre, normal


def body_parts():
    """Ribbon paths, thin strands and flower placements of the body.

    Eight ribbons rise from flat golden curls on the ground as a golden
    vortex. Two braid loosely up the main stem, one carries the second
    bloom, two sweep into curls holding flowers, and three peel away at
    golden heights carrying a flower at the tip and another on a sprig.
    """
    ribbons, tubes, flowers = [], [], []
    kmax = np.log(R0 * 0.82 / NECK_R) / np.log(PHI)
    roles = ["stem", "stem", "bloom2", "curl", "curl", "peel", "peel", "peel"]
    order = [0, 4, 2, 6, 1, 3, 5, 7]                 # spread roles round the base
    peel_k = iter([kmax / PHI ** 2, kmax / PHI, kmax / PHI ** 0.5])
    for i in range(N_RIBBONS):
        role = roles[order[i]]
        a0 = 2 * np.pi * i / N_RIBBONS + 0.4
        vpts, th_end = vortex(a0, R0 * 0.82, kmax, 1, n=120)
        root = ground_curl(vpts[0], vpts[0] - vpts[1], 8.0 + 2.0 * (i % 3), 1 if i % 2 else -1)
        w_root = np.linspace(1.6, 6.6, len(root)) ** 1.0
        pts = np.vstack([root, vpts[1:]])
        w = np.concatenate([w_root, np.linspace(6.6, 4.0, len(vpts) - 1)])

        if role == "peel":
            kp = next(peel_k)
            cut = int(kp / kmax * (len(vpts) - 1))
            pts = np.vstack([root, vpts[1:cut + 1]])
            w = w[:len(pts)]
            p0 = pts[-1]
            tan = pts[-1] - pts[-2]
            tan /= np.linalg.norm(tan)
            out = np.array([p0[0], p0[1], 0.0])
            out /= np.linalg.norm(out)
            tip = p0 + out * 13.0 + np.array([0, 0, 15.0])
            arc = bezier(p0, p0 + tan * 6 + [0, 0, 2], tip - out * 2 - [0, 0, 6], tip, n=45)[1:]
            pts = np.vstack([pts, arc])
            w = np.concatenate([w, np.linspace(w[-1], 2.6, len(arc))])
            axis = (tip - arc[-6]) / np.linalg.norm(tip - arc[-6]) + np.array([0, 0, 0.6])
            flowers.append((tip + 1.5 * axis / np.linalg.norm(axis), axis, 0))
            # a sprig off the arc carries a second, smaller flower
            m = arc[len(arc) // 2]
            side = np.cross(out, [0, 0, 1.0])
            sp_tip = m + out * 6.0 + side * (5.0 if i % 2 else -5.0) + np.array([0, 0, 4.0])
            sprig = bezier(m, m + out * 2 + [0, 0, 1], sp_tip - [0, 0, 2.5], sp_tip, n=24)
            ribbons.append((sprig, np.linspace(2.8, 2.2, len(sprig)), np.pi * 0.5, 0))
            sp_axis = (sp_tip - sprig[-5]) / np.linalg.norm(sp_tip - sprig[-5]) + [0, 0, 0.5]
            flowers.append((sp_tip + 1.2 * sp_axis / np.linalg.norm(sp_axis), sp_axis, 2))
        elif role == "stem":                     # a loose braid up the stem
            s = np.linspace(0, 1, 90)[1:]
            z = NECK_Z + (TIP[2] - 3.0 - NECK_Z) * s
            c = stem_axis(z)
            rr = NECK_R + (4.6 - NECK_R) * np.clip(s / 0.2, 0, 1)
            ang = th_end + 2 * np.pi * 0.7 * s
            up = c + np.column_stack([rr * np.cos(ang), rr * np.sin(ang), np.zeros_like(s)])
            pts = np.vstack([pts, up])
            w = np.concatenate([w, np.linspace(4.0, 4.6, len(s))])
        elif role == "bloom2":
            p0 = pts[-1]
            dirn = np.array([np.cos(th_end), np.sin(th_end), 0.0])
            up = bezier(p0, p0 + 9 * dirn + [0, 0, 6], BLOOM2_C - 9 * BLOOM2_AXIS + [-4, 0, 0],
                        BLOOM2_C - 1.5 * BLOOM2_AXIS)[1:]
            pts = np.vstack([pts, up])
            w = np.concatenate([w, np.linspace(4.0, 3.6, len(up))])
        else:                                    # sweep out into a golden curl
            p0 = pts[-1]
            side = np.array([np.cos(th_end), np.sin(th_end), 0.0])
            j = len([f for f in flowers if f[2] == 1])
            reach = p0 + side * 16 + np.array([0, 0, 22 + 6 * j])
            arc = bezier(p0, p0 + side * 6 + [0, 0, 10], reach - side * 4 - [0, 0, 2], reach)[1:]
            curl, centre, normal = golden_curl(arc[-1], arc[-1] - arc[-2], 8.5 - 1.5 * j)
            pts = np.vstack([pts, arc, curl])
            w = np.concatenate([w, np.linspace(4.0, 3.8, len(arc)),
                                np.linspace(3.8, 2.4, len(curl))])
            flowers.append((centre, normal, 1))
            tubes.append((np.vstack([curl[-1], centre]), np.array([1.0, 1.0])))
        ribbons.append((pts, w, np.pi * (0.5 + 0.25 * (i % 3)), len(root) - 3))

    # lattice core: 8 + 5 thin strands wound the other way inside the ribbons
    for count, sense in ((8, -1), (5, 1)):
        for i in range(count):
            spts, _ = vortex(2 * np.pi * i / count, R0 * 0.62, kmax, sense, n=100)
            tubes.append((spts, np.linspace(0.95, 0.75, len(spts))))
    # core column (hidden inside the vortex and braid)
    z = np.linspace(1.0, TIP[2] - 4.0, 60)
    rad = np.where(z < NECK_Z, NECK_R - 1.5 + 2.5 * (1 - z / NECK_Z) ** 2, 2.6)
    tubes.append((stem_axis(z), rad))
    return ribbons, tubes, flowers


def small_flower(pos, axis, size_class):
    """A small flame-petal plumeria (no lattice at this scale)."""
    R = BLOOM_R / PHI ** (3 if size_class == 0 else 4)
    if size_class == 1:
        R = BLOOM_R / PHI ** 3.3
    e1, e2, a = frame_from_axis(axis)
    petals = []
    for i in range(5):
        th = 0.4 + 2 * np.pi * i / 5
        W = R * 0.38
        base = np.array([1.0 * np.cos(th) - 0.3 * W * np.sin(th),
                         1.0 * np.sin(th) + 0.3 * W * np.cos(th), 0.0])
        petals.append(FlowPetal(base, th, np.radians(22), R, W, 0.35, 0.16, 0.25,
                                sweep=0.2, nest=0, scale_t=0.7, seed=int(pos[2] * 10) + i))
    return petals, (pos, e1, e2, a), R


def to_local(frame, x, y, z):
    c, e1, e2, a = frame
    qx, qy, qz = x - c[0], y - c[1], z - c[2]
    return (qx * e1[0] + qy * e1[1] + qz * e1[2], qx * e2[0] + qy * e2[1] + qz * e2[2],
            qx * a[0] + qy * a[1] + qz * a[2])


def local_box(frame, lo, hi):
    c, e1, e2, a = frame
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    world = c + corners @ np.vstack([e1, e2, a])
    return world.min(0), world.max(0)


def add_flower(vol, xs, ys, zs, petals, frame, R, bead_r, calyx_r):
    for p in petals:
        lo, hi = p.bbox()
        wlo, whi = local_box(frame, lo, hi)
        add(vol, xs, ys, zs, wlo, whi, lambda x, y, z, p=p: p.sdf(*to_local(frame, x, y, z)), k=0.5)

    def heart(x, y, z):
        lx, ly, lz = to_local(frame, x, y, z)
        bead = np.sqrt(lx ** 2 + ly ** 2 + ((lz - bead_r * 1.4) / 1.2) ** 2) - bead_r
        cal = np.sqrt(lx ** 2 + ly ** 2 + ((lz + calyx_r * 0.3) / 1.4) ** 2) - calyx_r
        return smin(bead, cal, 0.8)
    wlo, whi = local_box(frame, -np.full(3, calyx_r * 2.2), np.full(3, calyx_r * 2.2))
    add(vol, xs, ys, zs, wlo, whi, heart, k=0.8)


def stem_path(top, foot, n=24):
    """S-curved stem rising from `foot` to `top`, leaving and arriving
    vertically, so it never leans more than about 45 degrees."""
    h = top[2] - foot[2]
    return bezier(foot, foot + [0, 0, 0.42 * h], top - [0, 0, 0.42 * h], top, n=n)


def find_islands(vol, voxel, origin):
    """Lowest point of each part that would start printing in mid-air
    (nothing within one voxel on the layer below), merged when close."""
    from scipy import ndimage as ndi
    S = vol < 0
    k0 = int(np.ceil((BASE_CUT - origin[2]) / voxel)) + 2   # first layer sits on the plate
    found = []
    for k in range(k0, S.shape[2]):
        lab, n = ndi.label(S[:, :, k])
        if not n:
            continue
        near = ndi.binary_dilation(S[:, :, k - 1])
        hit = ndi.maximum(near, lab, np.arange(1, n + 1))
        for j in np.nonzero(hit == 0)[0] + 1:
            ii, jj = np.nonzero(lab == j)
            c = np.argmin((ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2)
            found.append(np.array([ii[c], jj[c], k]) * voxel + origin)
    merged = []
    for p in sorted(found, key=lambda q: q[2]):
        if all(np.linalg.norm(p - m) > 4.0 for m in merged):
            merged.append(p)
    return merged


def stems_for(islands, vol, voxel, origin):
    """A stem for each island, from the nearest solid below that it can
    reach without leaning past ~50 degrees (the plate, for the base).
    Stems may grow from earlier stems, so they branch like a vine."""
    from scipy import ndimage as ndi
    S = vol < 0
    surf = S & ~ndi.binary_erosion(S)
    pts = np.argwhere(surf) * voxel + np.asarray(origin)
    stems = []
    for p in sorted(islands, key=lambda q: -q[2]):     # highest first: the trunk
        top = p + [0, 0, 0.5]                      # bite into the island
        drop = top[2] - pts[:, 2]
        horiz = np.linalg.norm(pts[:, :2] - top[:2], axis=1)
        ok = (drop > 2.0) & (horiz < 0.7 * drop)
        cands = [(top[2] - BASE_CUT + 1.0, np.array([top[0], top[1], BASE_CUT - 1.0]))] \
            if top[2] < 15 else []
        if ok.any():
            d = np.sqrt(drop[ok] ** 2 + horiz[ok] ** 2)
            i = np.argmin(d)
            cands.append((d[i], pts[ok][i] - [0, 0, 0.6]))
        if not cands:
            print(f"  no stem target for island at {np.round(p, 1)}")
            continue
        foot = min(cands, key=lambda c: c[0])[1]
        path = stem_path(top, foot)
        stems.append((path, np.linspace(*STEM_R, len(path))))
        # later stems may branch off this one (below its top few mm)
        pts = np.vstack([pts, path[path[:, 2] < top[2] - 4.0] + [0, 0, 0.6]])
    return stems


def plan_stems(voxel=0.3, passes=8):
    """Grow stems until the body prints with no islands (coarse grid)."""
    stems = []
    for i in range(passes):
        vol, origin = build_body(voxel, stems, quiet=True)
        isl = find_islands(vol, voxel, origin)
        print(f"  stem pass {i + 1}: {len(isl)} island(s), {len(stems)} stem(s)")
        if not isl:
            break
        stems += stems_for(isl, vol, voxel, origin)
    return stems


def build_body(voxel, stems=(), quiet=False):
    say = (lambda *a: None) if quiet else print
    ribbons, tubes, flowers = body_parts()
    reach = BLOOM_R / PHI + 4                    # second bloom's petals
    lo = np.minimum([-R0 - 3, -R0 - 3, -0.6], BLOOM2_C - reach)
    hi = np.maximum([R0 + 3, R0 + 3, TIP[2] + 9], BLOOM2_C + reach)
    xs = np.arange(lo[0], hi[0], voxel, dtype=np.float32)
    ys = np.arange(lo[1], hi[1], voxel, dtype=np.float32)
    zs = np.arange(lo[2], hi[2], voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    t0 = time.time()

    # soft, flat-bottomed foot
    def foot(x, y, z):
        rr = np.sqrt(x ** 2 + y ** 2)
        d = soft_band(rr - R0 * 0.62, np.abs(z - 1.5) - 1.5, 5.0, 1.5)
        return np.maximum(d, -z)
    add(vol, xs, ys, zs, (-R0, -R0, -1), (R0, R0, 5), foot, k=0)

    for pts, w, twist, keep in ribbons:
        pts, w = smooth_path(pts, w, keep)
        segs = ribbon_sdf(pts, w, radial_ref, twist)
        for k0 in range(0, len(segs), 8):
            chunk = segs[k0:k0 + 8]
            P = np.array([s_[0] for s_ in chunk] + [chunk[-1][1]])
            pad = max(s_[4] for s_ in chunk) + RIB_T + 1.0
            add(vol, xs, ys, zs, P.min(0) - pad, P.max(0) + pad,
                lambda x, y, z, c=chunk: band_eval(c, x, y, z), k=0.8)
    say(f"  ribbons {time.time() - t0:.0f}s")

    for pts, rr in tubes:
        for k0 in range(0, len(pts) - 1, 10):
            seg, rad = pts[k0:k0 + 11], rr[k0:k0 + 11]
            pad = rad.max() + 1.0
            add(vol, xs, ys, zs, seg.min(0) - pad, seg.max(0) + pad,
                lambda x, y, z, s_=seg, r_=rad: capsule_chain(x, y, z, s_, r_), k=0.6)
    say(f"  lattice + stems {time.time() - t0:.0f}s")

    # second bloom: the main bloom at 1/phi, on the third ribbon
    e1, e2, a = frame_from_axis(BLOOM2_AXIS)
    frame2 = (BLOOM2_C, e1, e2, a)
    add_flower(vol, xs, ys, zs, bloom_petals(BLOOM_R / PHI, seed=7), frame2,
               BLOOM_R / PHI, 2.0, 4.6)
    say(f"  second bloom {time.time() - t0:.0f}s")

    for pos, axis, cls in flowers:
        petals, frame, R = small_flower(pos, axis, cls)
        add_flower(vol, xs, ys, zs, petals, frame, R, 1.1, 1.9)
    say(f"  small flowers {time.time() - t0:.0f}s")

    # stems that hold up whatever would otherwise start in mid-air
    for pts, rr in stems:
        add(vol, xs, ys, zs, pts.min(0) - 2.5, pts.max(0) + 2.5,
            lambda x, y, z, s_=pts, r_=rr: capsule_chain(x, y, z, s_, r_), k=0.8)

    # collar and keyed peg for the main bloom
    e1, e2, a = frame_from_axis(TIP_AXIS)
    ftip = (TIP, e1, e2, a)

    def collar(x, y, z):
        lx, ly, lz = to_local(ftip, x, y, z)
        rr = np.sqrt(lx ** 2 + ly ** 2)
        c = np.maximum(rr - 5.0 + 1.6 * np.clip(-lz / 4.0, 0, 1), np.maximum(lz, -5.0 - lz))
        return c
    add(vol, xs, ys, zs, TIP - 10, TIP + 10, collar, k=1.0)

    def peg(x, y, z):
        lx, ly, lz = to_local(ftip, x, y, z)
        return keyed_cylinder(lx, ly, lz, PEG_R, -0.5, PEG_H)
    add(vol, xs, ys, zs, TIP - 10, TIP + 10, peg, k=0)
    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])     # flat seat
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
        if part == "bloom":
            vol, origin = build_bloom(args.voxel)
        else:
            vol, origin = build_body(args.voxel, plan_stems())
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
