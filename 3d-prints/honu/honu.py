#!/usr/bin/env python3
"""Honu over coral (resin / MSLA, Saturn 4 Ultra), about 130 mm long.
One part, printed straight on the plate:

  base    a thin oval of rippled sand. The lobe-coral mound is hollow, open
          to the plate, and drains through two tunnels under the sand to
          the rim, so the cavity never seals on the plate.
  coral   Hawaiian reef corals: a lumpy lobe-coral mound (Porites lobata),
          a clump of blunt finger coral (Porites compressa) and a
          cauliflower-coral bouquet (Pocillopora), all pitted with polyp
          cups; a collector urchin (hāwaʻe) and a cowrie (leho) on the sand.
  honu    a green sea turtle banked into a glide. Its left front flipper
          sweeps down and rests its tip on a finger-coral knob; that is
          the only joint. Plated shell (vertebral, costal and marginal
          scutes), scaled head and flipper tops, ring-lidded eyes. The shell
          is hollow, with a drain and a vent hidden in the plastron.

Printability: everything grows upward from the plate or from the flipper
tip on the coral. Every surface texture is cut in, never raised; the
other three flippers rise away from the body; the neck lifts the head.

Usage
  python3 honu.py --voxel 0.3 --preview          # quick look
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
from bonsai_hibiscus import unit  # noqa: E402
from fractal_plumeria import capsule_chain, mesh_volume  # noqa: E402
from lattice_plumeria import BASE_CUT, add, bezier  # noqa: E402
from ohia_bonsai import add_tube, oval  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402

UP = np.array([0, 0, 1.0])

# ---------------------------------------------------------------- layout
BASE = (64.0, 39.0)                  # half-axes of the sand oval
RIM_H = 2.6                          # sand height at the rim
WALL = 2.2                           # wall of the hollow under the coral mound
LOBE = np.array([-30.0, 2.0])        # lobe-coral mound
FINGERS = np.array([4.0, 21.0])     # finger-coral clump
CAULI = np.array([-38.0, -24.0])     # cauliflower coral
URCHIN = np.array([44.0, -22.0])
COWRIE = np.array([16.0, -26.0])
# the honu: centre, heading, nose-down pitch and left-down roll. The pitch
# and roll put the body's lowest point where the left front flipper leaves
# the shell, so the whole turtle hangs from that flipper.
TURTLE = dict(C=np.array([10.0, -14.0, 54.0]), yaw=np.radians(18), pitch=np.radians(10), roll=np.radians(24))
TIP_LOCAL = np.array([8.0, 48.0, -6.0])              # left front flipper tip, turtle frame: out and a little down
REAR_TIP_LOCAL = np.array([-42.0, 27.0, -9.0])       # left rear flipper trails down onto the coral pinnacle
CAULI2 = np.array([34.0, 12.0])                      # a second, smaller cauliflower clump
FINGERS2 = np.array([-50.0, 17.0])                   # a small finger clump at the back
KNOLL = np.array([28.0, -8.0])                       # a small lobe-coral knoll


# -------------------------------------------------------------- helpers

def ell(x, y, z, rx, ry, rz):
    """Ellipsoid distance (good near the surface)."""
    k0 = np.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
    k1 = np.sqrt((x / rx ** 2) ** 2 + (y / ry ** 2) ** 2 + (z / rz ** 2) ** 2)
    return np.where(k1 > 1e-9, k0 * (k0 - 1) / np.maximum(k1, 1e-9), -np.minimum(np.minimum(rx, ry), rz))


def ell2(v, w, c, t):
    k0 = np.sqrt((v / c) ** 2 + (w / t) ** 2)
    k1 = np.sqrt((v / c ** 2) ** 2 + (w / t ** 2) ** 2)
    return np.where(k1 > 1e-9, k0 * (k0 - 1) / np.maximum(k1, 1e-9), -np.minimum(c, t))


def seg_dist(px, py, segs):
    """Distance in 2D from points to a set of segments (n, 2, 2)."""
    best = np.full(px.shape, 1e9)
    for (ax, ay), (bx, by) in segs:
        dx, dy = bx - ax, by - ay
        h = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-12), 0, 1)
        best = np.minimum(best, np.hypot(px - ax - h * dx, py - ay - h * dy))
    return best


def polyline(pts, closed=False):
    pts = np.asarray(pts, float)
    out = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    if closed:
        out.append((pts[-1], pts[0]))
    return out


def rot(yaw, pitch, roll):
    """Local -> world: roll about x (left side down for +roll), pitch about y
    (nose down for +pitch), then yaw about z."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


CONTACT = TURTLE["C"] + rot(TURTLE["yaw"], TURTLE["pitch"], TURTLE["roll"]) @ TIP_LOCAL   # where the flipper tip rests
CONTACT2 = TURTLE["C"] + rot(TURTLE["yaw"], TURTLE["pitch"], TURTLE["roll"]) @ REAR_TIP_LOCAL   # rear flipper on the pinnacle


# ------------------------------------------------------------------ sand

def sand(x, y):
    """Sand height: rim height, a rise under the coral, wind-free ripples."""
    r = np.sqrt(((x - LOBE[0]) / 34.0) ** 2 + ((y - LOBE[1]) / 24.0) ** 2)
    rip = 0.42 * np.sin(0.62 * (0.93 * x + 0.37 * y) + 0.9 * np.sin(0.09 * y + 0.05 * x)) \
        * (0.75 + 0.25 * np.sin(0.21 * y - 0.08 * x))
    fade = np.clip(r - 0.6, 0, 1)                       # ripples die out under the coral
    return RIM_H + 1.0 + 2.6 * np.exp(-r ** 2 * 1.5) + rip * fade


def base_sdf(x, y, z, open_bottom=False):
    flare = 1.5 * np.clip((RIM_H - z) / RIM_H, 0, 1)
    d = np.maximum(oval(x, y, BASE[0] + flare, BASE[1] + flare), z - sand(x, y))
    return d if open_bottom else np.maximum(d, -z)


# ----------------------------------------------------------------- coral

def lobe_blobs():
    """Lumps of the lobe-coral mound: (centre, radii). Each lump's centre is
    high enough above the next lump down that its underside stays buried."""
    rng = np.random.default_rng(5)
    main_c, main_r = np.array([LOBE[0], LOBE[1], 5.0]), np.array([22.0, 16.0, 14.0])
    out = [(main_c, main_r)]
    for i in range(18):                                   # lumps sunk into the dome along its normal
        zz = 0.15 + 0.85 * (i + 0.5) / 18
        a = 2.39996 * i
        q = np.array([np.sqrt(1 - zz ** 2) * np.cos(a), np.sqrt(1 - zz ** 2) * np.sin(a), zz])
        s_ = main_c + q * main_r
        n_ = unit(q / main_r)
        rad = rng.uniform(5.5, 8.5)
        out.append((s_ - n_ * rad * 0.5, np.array([rad, rad, rad * 0.85])))
    return out


def finger_tubes():
    """Blunt finger coral: thick upright fingers that fork, plus the knob
    the honu's flipper rests on."""
    rng = np.random.default_rng(11)
    tubes = []
    gz = sand(*FINGERS) - 2.0

    def finger(p0, d, L, r0, r1, n=14):
        d = unit(d)
        pts = bezier(p0, p0 + 0.35 * L * d, p0 + 0.7 * L * unit(d + [0, 0, 0.5]), p0 + L * unit(d + [0, 0, 0.35]), n=n)
        return pts, np.linspace(r0, r1, n)
    # the tall finger under the flipper tip, ending in a broad knob
    top = CONTACT - [0, 0, 2.2]
    base = np.array([CONTACT[0] + 3.0, CONTACT[1] - 5.0, gz])
    pts = bezier(base, base + [0, 0, 14], top - [2.0, 1.0, 12], top, n=20)
    tubes.append((pts, np.linspace(3.4, 3.0, 20)))
    tubes.append((np.array([top - [0, 0, 1.5], top + [0, 0, 0.3]]), np.array([3.8, 3.5])))
    for i in range(14):
        a = 2.39996 * i + 0.4
        rr = 2.0 + 10.0 * np.sqrt((i + 0.5) / 14)
        p0 = np.array([FINGERS[0] + rr * np.cos(a), FINGERS[1] + rr * 0.8 * np.sin(a), gz])
        out_ = np.array([np.cos(a), np.sin(a), 0.0])
        tilt = rng.uniform(0.1, 0.6)
        d = unit(out_ * tilt + rng.normal(0, 0.12, 3) * [1, 1, 0] + UP)
        L = rng.uniform(13, 30) * (1.2 - 0.03 * rr)
        r0 = rng.uniform(2.3, 2.9)
        p, r = finger(p0, d, L, r0, r0 * rng.uniform(0.82, 0.95))
        tubes.append((p, r))
        for frac in (0.4, 0.62):                          # forks, rising steeply
            if L > 15 and rng.uniform() < 0.55:
                k = int(frac * len(p))
                b = rng.uniform(0, 2 * np.pi)
                d2 = unit(np.array([np.cos(b), np.sin(b), 0]) * rng.uniform(0.3, 0.7) + UP * 1.3)
                q0 = p[k] - d2 * r[k] * 0.8
                tubes.append(finger(q0, d2, L * rng.uniform(0.35, 0.6), r[k] * 0.85, r[k] * 0.75, n=10))
    return tubes


def small_fingers(where, n, seed):
    """A small clump of blunt finger coral."""
    rng = np.random.default_rng(seed)
    gz = sand(*where) - 2.0
    out = []
    for i in range(n):
        a = 2.39996 * i + rng.uniform(0, 1)
        rr = 1.5 + 4.5 * np.sqrt((i + 0.5) / n)
        p0 = np.array([where[0] + rr * np.cos(a), where[1] + rr * np.sin(a), gz])
        d = unit(np.array([np.cos(a), np.sin(a), 0]) * rng.uniform(0.1, 0.5) + UP)
        L = rng.uniform(9, 18)
        r0 = rng.uniform(2.1, 2.6)
        pts = bezier(p0, p0 + 0.35 * L * d, p0 + 0.7 * L * unit(d + [0, 0, 0.5]), p0 + L * unit(d + [0, 0, 0.35]), n=12)
        out.append((pts, np.linspace(r0, r0 * 0.88, 12)))
    return out


def pinnacle():
    """A lobe-coral pinnacle (bommie) rising off the mound to meet the honu's
    rear flipper: a tapering knobbly column. Lumps sit on its upper flanks
    only, sunk halfway, so each one's underside stays inside the column."""
    rng = np.random.default_rng(23)
    top = CONTACT2 - [0, 0, 2.6]
    base = np.array([top[0] + 3.0, top[1] + 4.0, sand(top[0] + 3.0, top[1] + 4.0) - 1.0])
    pts = bezier(base, base + [0, 0, 12], top - [1.5, 2.0, 12], top, n=24)
    rad = np.linspace(9.0, 4.4, 24)
    blobs = []
    for i in range(4, 24, 2):
        for _ in range(2):
            a = rng.uniform(0, 2 * np.pi)
            el = np.radians(rng.uniform(40, 75))
            dr = np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)])
            rl = rad[i] * rng.uniform(0.45, 0.6)
            blobs.append((pts[i] + dr * (rad[i] - 0.6 * rl), np.array([rl, rl, rl * 0.9])))
    blobs.append((top + [0, 0, 0.6], np.array([4.4, 4.4, 3.6])))   # the knob the flipper rests on
    return (pts, rad), blobs


def knoll_blobs():
    c = np.array([KNOLL[0], KNOLL[1], sand(*KNOLL) - 3.0])
    out = [(c, np.array([9.0, 7.5, 7.0]))]
    for i in range(7):
        a = 2.39996 * i
        zz = 0.3 + 0.6 * (i + 0.5) / 7
        q = np.array([np.sqrt(1 - zz ** 2) * np.cos(a), np.sqrt(1 - zz ** 2) * np.sin(a), zz])
        rl = 3.2 + 0.5 * np.sin(3 * i)
        out.append((c + q * [9.0, 7.5, 7.0] - unit(q / [9.0, 7.5, 7.0]) * rl * 0.5, np.array([rl, rl, rl * 0.85])))
    return out


def cauli_tubes(where=CAULI, seed=17, sc=1.0):
    """Cauliflower coral: a bouquet of short branches splaying up and out,
    each ending in a blunt flattened lobe."""
    rng = np.random.default_rng(seed)
    tubes = []
    gz = sand(*where) - 1.5
    root = np.array([where[0], where[1], gz])
    for i in range(34):
        a = 2.39996 * i
        el = np.radians(30 + 58 * ((i + 0.5) / 34) ** 0.7)   # outer branches splay low, inner ones climb
        d = np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)])
        L = rng.uniform(8, 12) * (1.25 - 0.4 * np.cos(el)) * sc
        p0 = root + [rng.normal(0, 1.5), rng.normal(0, 1.5), 0]
        pts = bezier(p0, p0 + 0.4 * L * d, p0 + 0.8 * L * unit(d + UP * 0.4), p0 + L * unit(d + UP * 0.6), n=10)
        tubes.append((pts, np.linspace(2.2, 1.8, 10)))
        tip = pts[-1]
        for j in range(2):                               # knobbly lobes at the tip, rising
            b = a + 2.6 * j + rng.uniform(-0.5, 0.5)
            q = tip + 1.3 * np.array([np.cos(b), np.sin(b), 0]) + [0, 0, 1.0]
            tubes.append((np.array([tip - [0, 0, 0.5], q]), np.array([1.75, 1.4])))
    return tubes


def corallite_pits(x, y, z, d, n, s=1.6, r=0.42, depth=0.4):
    """Cut a scatter of polyp cups into the surface (a jittered lattice).
    Each cup is a short column along the surface normal n, so it always
    opens at the surface; none are cut where the surface overhangs. Pits
    keep a wall of at least 0.26 mm between them (thinner walls crumble)."""
    q = np.stack([x / s, y / s + 0.5 * np.floor(x / s), z / s + 0.5 * np.floor(y / s)], -1)
    i = np.floor(q)
    h = np.sin(i[..., 0] * 12.9898 + i[..., 1] * 78.233 + i[..., 2] * 37.719) * 43758.5453
    j = (h - np.floor(h) - 0.5) * 0.18                  # small jitter: pits never meet across a cell wall
    f = q - i - 0.5 - np.stack([j, np.roll(j, 0) * -0.8, j * 0.6], -1)
    along = np.einsum("...i,...i->...", f, n)
    cell = np.sqrt(np.maximum(np.einsum("...i,...i->...", f, f) - along ** 2, 0)) * s
    wgt = np.clip((n[..., 2] + 0.35) / 0.3, 0, 1)      # fade out on overhangs
    return np.maximum(d, np.minimum(r - cell, d + depth * wgt))


def urchin_sdf():
    """Collector urchin: a squat test with stubby spines on its upper half
    only, every spine climbing at least 20 degrees."""
    gz = sand(*URCHIN)
    c = np.array([URCHIN[0], URCHIN[1], gz + 2.6])
    spines = []
    n = 70
    for i in range(n):
        zz = 1 - (i + 0.5) / n
        el = np.arcsin(zz)
        if el < np.radians(20):
            continue
        a = 2.39996 * i
        dr = np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)])
        p = c + dr * np.array([5.0, 5.0, 3.6]) * 0.85
        spines.append((p, p + dr * 2.4))

    def f(x, y, z):
        d = ell(x - c[0], y - c[1], z - c[2], 6.0, 6.0, 4.2)
        for a, b in spines:
            ab = b - a
            h = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1] + (z - a[2]) * ab[2]) / (ab @ ab), 0, 1)
            dd = np.sqrt((x - a[0] - h * ab[0]) ** 2 + (y - a[1] - h * ab[1]) ** 2 + (z - a[2] - h * ab[2]) ** 2)
            d = np.minimum(d, dd - (0.62 - 0.12 * h))
        return d
    return f, (c - [9, 9, 4.5], c + [9, 9, 8])


def cowrie_sdf():
    gz = sand(*COWRIE)
    c = np.array([COWRIE[0], COWRIE[1], gz + 0.6])
    a = np.radians(30)
    ca, sa = np.cos(a), np.sin(a)

    def f(x, y, z):
        u = (x - c[0]) * ca + (y - c[1]) * sa
        v = -(x - c[0]) * sa + (y - c[1]) * ca
        w = z - c[2]
        d = ell(u, v, w, 7.5, 4.8, 3.8)
        ridge = 0.25 * np.cos(np.pi * u / 7.5) * np.exp(-(v / 1.5) ** 2)   # a soft dorsal line
        return d - ridge
    return f, (c - [9, 9, 4], c + [9, 9, 5])


# ------------------------------------------------------------------ honu

A_SH, B_SH, TOP_SH, BOT_SH = 35.0, 28.5, 12.0, 6.0     # shell half-length, half-width, dome, plastron
T_WALL = 2.0


def shell_sdf(x, y, z, off=0.0):
    """Carapace and plastron: a heart-shaped dome, narrowing behind."""
    by = B_SH * (1 - 0.24 * np.clip(-x / A_SH, 0, 1) ** 1.4) * (1 - 0.06 * np.clip(x / A_SH, 0, 1) ** 2)
    rz = np.where(z > 0, TOP_SH * (1 - 0.15 * np.clip(-x / A_SH, 0, 1)), BOT_SH)
    return ell(x, y, z, A_SH - off, by - off, rz - off)


def scute_segments():
    s = 1.0
    segs = []
    vert = [(29, 0), (27, 5), (20, 9.5), (14, 8.5), (7, 10), (1, 9), (-5, 10), (-12, 9), (-18, 9.5), (-24, 8), (-29, 4), (-31, 0)]
    for side in (1, -1):
        segs += polyline([(x_, side * y_) for x_, y_ in vert])
    for xb, yb in ((14, 8.5), (1, 9), (-12, 9), (-24, 8)):
        segs.append((np.array([xb, -yb]), np.array([xb, yb])))
    for side in (1, -1):                                  # costal boundaries
        for (x0, y0), (x1, y1) in (((20, 9.5), (24, 21)), ((7, 10), (7, 24.5)), ((-5, 10), (-8, 24)), ((-18, 9.5), (-22, 18))):
            segs.append((np.array([x0, side * y0]), np.array([x1, side * y1])))
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = []
    for tt in t:                                          # marginal ring
        x_ = 0.86 * A_SH * np.cos(tt)
        by = B_SH * (1 - 0.24 * np.clip(-x_ / A_SH, 0, 1) ** 1.4) * (1 - 0.06 * np.clip(x_ / A_SH, 0, 1) ** 2)
        ring.append((x_, 0.84 * by * np.sin(tt)))
    segs += polyline(ring, closed=True)
    for tt in np.linspace(0, 2 * np.pi, 25, endpoint=False):   # marginal scutes
        if abs(np.cos(tt)) > 0.97 and np.cos(tt) > 0:
            continue
        x_ = A_SH * np.cos(tt)
        by = B_SH * (1 - 0.24 * np.clip(-x_ / A_SH, 0, 1) ** 1.4) * (1 - 0.06 * np.clip(x_ / A_SH, 0, 1) ** 2)
        segs.append((np.array([0.84 * x_, 0.82 * by * np.sin(tt)]), np.array([1.05 * x_, 1.05 * by * np.sin(tt)])))
    return [(np.asarray(a) * s, np.asarray(b) * s) for a, b in segs]


def paddle(spine, chord, thick, up_hint=UP):
    """A flipper: a flat paddle along a spine, elliptical in section.
    Returns (fn, frames); fn gives (distance, span position, chord offset,
    thickness offset) so the top can be scaled."""
    n = len(spine)
    T = np.gradient(spine, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    W = up_hint - (T @ up_hint)[:, None] * T
    W /= np.linalg.norm(W, axis=1)[:, None]
    V = np.cross(W, T)
    seglen = np.linalg.norm(np.diff(spine, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)])

    def fn(P):
        best = np.full(P.shape[0], 1e9)
        s_b, v_b, w_b = np.zeros_like(best), np.zeros_like(best), np.zeros_like(best)
        for i in range(n - 1):
            a, b = spine[i], spine[i + 1]
            ab = b - a
            h = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
            dvec = P - (a + h[:, None] * ab)
            Vi = (1 - h)[:, None] * V[i] + h[:, None] * V[i + 1]
            Wi = (1 - h)[:, None] * W[i] + h[:, None] * W[i + 1]
            v = np.einsum("ij,ij->i", dvec, Vi)
            w = np.einsum("ij,ij->i", dvec, Wi)
            u = dvec @ (ab / np.linalg.norm(ab))
            c = (1 - h) * chord[i] + h * chord[i + 1]
            t = (1 - h) * thick[i] + h * thick[i + 1]
            e = ell2(v, w, c, t)
            # past the ends: a rounded, ellipsoidal cap rather than a flat cut
            dd = np.where(np.abs(u) > 1e-6, ell(v, w, u, c, t, np.maximum(t, 0.8 * c)), e)
            m = dd < best
            best = np.where(m, dd, best)
            s_b = np.where(m, cum[i] + h * seglen[i], s_b)
            v_b = np.where(m, v, v_b)
            w_b = np.where(m, w, w_b)
        return best, s_b, v_b, w_b
    lo, hi = spine.min(0) - chord.max() - 1, spine.max(0) + chord.max() + 1
    return fn, (lo, hi), cum[-1]


def flipper_profile(n, L, cmax, t0, t_tip, front=True):
    s = np.linspace(0, 1, n)
    if front:      # narrow wrist, broad blade a third of the way out, long pointed taper
        c = cmax * (0.45 + 0.55 * np.sin(np.pi / 2 * np.clip(s / 0.38, 0, 1))) * (1 - 0.86 * np.clip((s - 0.38) / 0.62, 0, 1) ** 1.5)
    else:          # rear flippers: rounded paddles
        c = cmax * (0.5 + 0.5 * np.sin(np.pi / 2 * np.clip(s / 0.7, 0, 1)))      # widening fan
        c = c * np.sqrt(np.clip(1 - (np.clip(s - 0.7, 0, 1) / 0.32) ** 2, 0.02, 1))   # rounded end
    t = t0 + (t_tip - t0) * s
    return np.maximum(c, t + 0.2), t


def honu(C, yaw, pitch, roll, contact):
    """The turtle as (fn, (lo, hi)). Local frame: x forward, y left, z up."""
    R = rot(yaw, pitch, roll)
    tgt = R.T @ (contact - C)                             # flipper tip, local
    flippers = []
    # left front: the downstroke onto the coral
    p0 = np.array([15.0, 17.0, -1.5])
    sp = bezier(p0, p0 + [0, 10, -2.5], p0 + 0.62 * (tgt - p0) + [5, 0, 1], tgt, n=22)
    L = np.sum(np.linalg.norm(np.diff(sp, axis=0), axis=1))
    c, t = flipper_profile(22, L, 7.6, 1.9, 1.35)
    flippers.append(paddle(sp, c, t))
    # right front: high in the upstroke, swept back
    p0 = np.array([15.0, -18.0, -1.0])
    tip = np.array([-10.0, -53.0, 4.0])
    sp = bezier(p0, p0 + [2, -14, 2], p0 + [1, -30, 4.5], tip, n=22)
    c, t = flipper_profile(22, 0, 7.6, 1.9, 0.9)
    flippers.append(paddle(sp, c, t))
    # rear flippers, trailing and lifted
    for side in (1, -1):
        p0 = np.array([-25.0, side * 13.0, -1.5])
        if side == 1:                                     # the low one trails down onto the pinnacle
            tip = REAR_TIP_LOCAL
            sp = bezier(p0, p0 + [-6, 4, -1.5], tip + [5, -2, 2.5], tip, n=12)
        else:
            tip = np.array([-44.0, -25.0, 5.5])
            sp = bezier(p0, p0 + [-6, -4, 1], tip + [4, 1, -1.5], tip, n=12)
        c, t = flipper_profile(12, 0, 6.0, 1.7, 1.1, front=False)
        flippers.append(paddle(sp, c, t))
    scutes = scute_segments()
    # head and neck: the neck lifts the head against the body's pitch
    neck_up = np.radians(22)
    Rn = np.array([[np.cos(neck_up), 0, -np.sin(neck_up)], [0, 1, 0], [np.sin(neck_up), 0, np.cos(neck_up)]])
    n0 = np.array([25.0, 0, 1.5])
    hc = n0 + Rn @ np.array([17.0, 0, 0])
    neck = bezier(n0, n0 + Rn @ [6, 0, 0], n0 + Rn @ [12, 0, 0], hc - Rn @ [3, 0, 1.0], n=8)
    neck_r = np.linspace(5.6, 4.8, 8)
    head_segs = polyline([(8, 0), (3.5, 2.3), (-1, 2.8), (-4.5, 2.1)]) + polyline([(8, 0), (3.5, -2.3), (-1, -2.8), (-4.5, -2.1)]) \
        + [(np.array([3.5, -2.3]), np.array([3.5, 2.3])), (np.array([-1.0, -2.8]), np.array([-1.0, 2.8])),
           (np.array([-4.5, -2.1]), np.array([-4.5, 2.1])), (np.array([8.0, 0.0]), np.array([-1.0, 0.0]))] \
        + polyline([(-1, 2.8), (-3, 4.8)]) + polyline([(-1, -2.8), (-3, -4.8)])
    eyes = [np.array([2.6, s_ * 5.0, 2.0]) for s_ in (1, -1)]
    holes = [np.array([12.0, 12.0]), np.array([-10.0, -12.0])]   # drain (low side) and vent

    def fn(x, y, z):
        sh = x.shape
        P = np.stack([x.ravel() - C[0], y.ravel() - C[1], z.ravel() - C[2]], -1) @ R
        lx, ly, lz = P[:, 0], P[:, 1], P[:, 2]
        d = shell_sdf(lx, ly, lz)
        # scutes: grooves on the carapace only
        band = (np.abs(d) < 1.0) & (lz > -0.8)
        if band.any():                                    # groove depth fades out toward the rim
            g = seg_dist(lx[band], ly[band], scutes)
            wgt = np.clip((lz[band] + 0.8) / 1.2, 0, 1)
            d[band] = np.maximum(d[band], np.minimum(0.42 - g, d[band] + 0.55 * wgt))
        # neck and head
        dn = np.full_like(d, 1e9)
        for i in range(len(neck) - 1):
            a, b = neck[i], neck[i + 1]
            ab = b - a
            h = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
            rr = neck_r[i] + h * (neck_r[i + 1] - neck_r[i])
            dn = np.minimum(dn, np.linalg.norm(P - a - h[:, None] * ab, axis=1) - rr)
        H = (P - hc) @ Rn                                 # head frame
        hx, hy, hz = H[:, 0], H[:, 1], H[:, 2]
        dh = ell(hx, hy, hz + 0.4 * np.clip(hx / 9, 0, 1) ** 2, 9.5, 6.6, 5.8)
        dh = smin(dh, ell(hx - 6.0, hy, hz + 1.0, 4.4, 4.4, 4.0), 1.8)   # blunt snout
        hb = (np.abs(dh) < 1.0)
        if hb.any():                                      # scales on top, mouth line on the sides
            top = hb & (hz > 0.5)
            g = seg_dist(hx[top], hy[top], head_segs)
            wgt = np.clip((hz[top] - 0.5) / 1.0, 0, 1)
            dh[top] = np.maximum(dh[top], np.minimum(0.32 - g, dh[top] + 0.4 * wgt))
            side = hb & (hx > -5)
            mouth = np.abs(hz[side] + 1.6 + 0.06 * (hx[side] - 4) ** 2 * (hx[side] < 4))
            wgt = np.clip((hx[side] + 5) / 1.5, 0, 1)
            dh[side] = np.maximum(dh[side], np.minimum(0.3 - mouth, dh[side] + 0.45 * wgt))
        for e in eyes:                                    # lidded eye: a bulge ringed by a groove
            q = H - e
            de = np.linalg.norm(q, axis=1)
            dh = smin(dh, de - 1.6, 0.6)
            ring = np.abs(de - 2.0)
            dh = np.maximum(dh, np.minimum(0.35 - ring, dh + 0.5))
        d = smin(d, smin(dn, dh, 2.5), 3.0)
        # flippers
        for f_, (blo, bhi), _ in flippers:
            m = np.all((P > blo) & (P < bhi), axis=1)
            if not m.any():
                continue
            df, s_, v_, w_ = f_(P[m])
            wgt = np.clip(w_ / 0.6, 0, 1) * np.clip((s_ - 5.0) / 2.0, 0, 1)   # scales on the upper side only
            gs = np.minimum(np.abs((s_ / 3.4) - np.round(s_ / 3.4)) * 3.4,
                            np.abs(((v_ + 1.7 * (np.floor(s_ / 3.4) % 2)) / 3.4) - np.round((v_ + 1.7 * (np.floor(s_ / 3.4) % 2)) / 3.4)) * 3.4)
            df = np.maximum(df, np.minimum(0.28 - gs, df + 0.35 * wgt))
            d[m] = smin(d[m], df, 2.0)
        # tail
        tl = ell(lx + 35.0, ly, lz + 1.0, 6.0, 2.8, 2.2)
        d = smin(d, tl, 1.5)
        # hollow shell, with a drain and a vent through the plastron
        inner = shell_sdf(lx, ly, lz, T_WALL)
        d = np.maximum(d, -inner)
        for hx_, hy_ in holes:
            cyl = np.maximum(np.hypot(lx - hx_, ly - hy_) - 1.2, np.maximum(lz - 0.0, -lz - 12.0))
            d = np.maximum(d, -cyl)
        return d.reshape(sh)
    # world bounding box
    pts = [np.array([a_, b_, c_]) for a_ in (-50, 62) for b_ in (-34, 34) for c_ in (-9, 16)]
    for _, (blo, bhi), _ in flippers:
        pts += [np.array([a_, b_, c_]) for a_ in (blo[0], bhi[0]) for b_ in (blo[1], bhi[1]) for c_ in (blo[2], bhi[2])]
    W_ = np.array([C + R @ p for p in pts])
    return fn, (W_.min(0) - 1.0, W_.max(0) + 1.0), tgt


# ------------------------------------------------------------------ build

def bounds():
    """The base plus the whole turtle: its raised flipper reaches past the
    edge of the sand."""
    _, (tlo, thi), _ = honu(TURTLE["C"], TURTLE["yaw"], TURTLE["pitch"], TURTLE["roll"], CONTACT)
    lo = np.minimum([-BASE[0] - 3, -BASE[1] - 3, -0.6], tlo)
    hi = np.maximum([BASE[0] + 3, BASE[1] + 3, 10.0], thi)
    lo[2] = -0.6
    return lo, hi


def build(voxel, stems=(), quiet=False, region=None):
    say = (lambda *a: None) if quiet else print
    t0 = time.time()
    lo, hi = bounds() if region is None else (np.asarray(region[0]), np.asarray(region[1]))
    xs = np.arange(lo[0], hi[0], voxel, dtype=np.float32)
    ys = np.arange(lo[1], hi[1], voxel, dtype=np.float32)
    zs = np.arange(lo[2], hi[2], voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    blo, bhi = (-BASE[0] - 3, -BASE[1] - 3, -1), (BASE[0] + 3, BASE[1] + 3, RIM_H + 6)
    add(vol, xs, ys, zs, blo, bhi, base_sdf, k=0, chunk=8)
    say(f"  sand {time.time() - t0:.0f}s")
    # coral, built in its own buffer so the polyp pits stay on the coral
    clo, chi = np.array([-63.0, -38.0, 0.0]), np.array([46.0, 37.0, max(CONTACT[2], CONTACT2[2]) + 3])
    sl = lp.sub(xs, ys, zs, clo, chi)
    if all(s_.stop > s_.start for s_ in sl):
        cx, cy, cz = xs[sl[0]], ys[sl[1]], zs[sl[2]]
        cv = np.full((len(cx), len(cy), len(cz)), 3.0, np.float32)
        (ppts, prad), pblobs = pinnacle()
        add_tube(cv, cx, cy, cz, ppts, prad, k=0)
        for c, r in lobe_blobs() + knoll_blobs() + pblobs:
            add(cv, cx, cy, cz, c - r - 3, c + r + 3,           # pad past the blend, or its fillet is cut flat
                lambda x, y, z, c=c, r=r: ell(x - c[0], y - c[1], z - c[2], *r), k=2.0, chunk=16)
        more = small_fingers(FINGERS2, 7, 31) + cauli_tubes(CAULI2, 41, 0.75)
        for pts, rr in finger_tubes() + cauli_tubes() + more:   # plain union: a blend webs leaning
            add_tube(cv, cx, cy, cz, pts, np.asarray(rr, float), k=0)   # fingers together in mid-air

        # carve the polyp cups slab by slab
        idx = np.argwhere(np.abs(cv) < 1.0)                 # near the surface
        idx = idx[np.all((idx > 0) & (idx < np.array(cv.shape) - 1), axis=1)]
        for c0 in range(0, len(idx), 4_000_000):
            ii = idx[c0:c0 + 4_000_000]
            a_, b_, c_ = ii[:, 0], ii[:, 1], ii[:, 2]
            g = np.stack([cv[a_ + 1, b_, c_] - cv[a_ - 1, b_, c_], cv[a_, b_ + 1, c_] - cv[a_, b_ - 1, c_],
                          cv[a_, b_, c_ + 1] - cv[a_, b_, c_ - 1]], -1)
            g /= np.linalg.norm(g, axis=1)[:, None] + 1e-9
            new = corallite_pits(cx[a_], cy[b_], cz[c_], cv[a_, b_, c_], g)
            cv[a_, b_, c_] = new
        vol[sl] = smin(vol[sl], cv, 1.2)
        del cv
    say(f"  coral {time.time() - t0:.0f}s")
    for f, (a, b) in (urchin_sdf(), cowrie_sdf()):
        add(vol, xs, ys, zs, a, b, f, k=0.8, chunk=16)
    # hollow the lobe mound: a domed cavity open to the plate (its roof
    # closes inward, so it needs no support), drained by two tunnels with
    # 45-degree pitched roofs running under the sand to the rim
    mc, mr = lobe_blobs()[0]
    cav_r = mr - WALL - 0.4
    tunnels = [(np.array([mc[0], mc[1]]), np.array([-BASE[0] - 3, mc[1] + 4.0])),
               (np.array([mc[0], mc[1]]), np.array([mc[0] - 6.0, -BASE[1] - 3]))]

    (ppts, prad), _ = pinnacle()

    def cavity(x, y, z):
        d = ell(x - mc[0], y - mc[1], z - mc[2], *cav_r)
        # the pinnacle's column stays solid through the cavity, down to the plate
        k_ = 0
        ax = np.hypot(x - ppts[k_, 0], y - ppts[k_, 1])
        d = np.maximum(d, -(ax - prad[0] + 0.5))
        for p, q in tunnels:
            ab = q - p
            h = np.clip(((x - p[0]) * ab[0] + (y - p[1]) * ab[1]) / (ab @ ab), 0, 1)
            lat = np.hypot(x - p[0] - h * ab[0], y - p[1] - h * ab[1])
            d = np.minimum(d, z + lat - 2.6)             # pitched roof: apex 2.6 mm, sides at 45 degrees
        return -d
    clo2, chi2 = mc - cav_r - [40, 45, 1], mc + cav_r + 1
    sl = lp.sub(xs, ys, zs, clo2, chi2)
    for i in range(sl[0].start, sl[0].stop, 8):
        sx = slice(i, min(i + 8, sl[0].stop))
        X, Y, Z = np.meshgrid(xs[sx], ys[sl[1]], zs[sl[2]], indexing="ij")
        vol[sx, sl[1], sl[2]] = np.maximum(vol[sx, sl[1], sl[2]], cavity(X, Y, Z))
    say(f"  hollow mound {time.time() - t0:.0f}s")
    f, (a, b), _ = honu(TURTLE["C"], TURTLE["yaw"], TURTLE["pitch"], TURTLE["roll"], CONTACT)
    add(vol, xs, ys, zs, a, b, f, k=1.0, chunk=4)
    say(f"  honu {time.time() - t0:.0f}s")
    for pts, rr in stems:
        add(vol, xs, ys, zs, pts.min(0) - 2.5, pts.max(0) + 2.5,
            lambda x, y, z, s_=pts, r_=rr: capsule_chain(x, y, z, s_, r_), k=0.8)
    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])
    fill_voids(vol)
    drop_specks(vol, voxel, say)
    n = trim_crumbs(vol, voxel, zs)
    say(f"  sealed voids filled, specks dropped, {n} crumbs trimmed {time.time() - t0:.0f}s")
    return vol, (xs[0], ys[0], zs[0])


def trim_crumbs(vol, voxel, zs, max_area=0.06):
    """Texture cut into a sloping underside can leave specks a voxel or two
    across that start in mid-air. Working up from the plate, remove any
    piece of a layer under max_area (mm^2) that has nothing under it; the
    layer above is then checked against the cleaned one. Larger pieces are
    left alone, so a real island still shows up in the island check."""
    from scipy import ndimage as ndi
    max_px = max(1, int(max_area / voxel ** 2))
    k0 = int(np.searchsorted(zs, BASE_CUT)) + 2
    n_cut = 0
    for k in range(max(k0, 1), vol.shape[2]):
        S = vol[:, :, k] < 0
        if not S.any():
            continue
        lab, n = ndi.label(S)
        below = ndi.binary_dilation(vol[:, :, k - 1] < 0)
        hit = ndi.maximum(below, lab, np.arange(1, n + 1))
        size = ndi.sum(S, lab, np.arange(1, n + 1))
        bad = np.nonzero((hit == 0) & (size <= max_px))[0] + 1
        if len(bad):
            m = np.isin(lab, bad)
            vol[:, :, k][m] = 0.05
            n_cut += len(bad)
    return n_cut


def drop_specks(vol, voxel, say=print):
    """Remove crumbs of solid floating free (texture cuts can leave a few
    where blends web two parts together). Keeps the main body and anything
    reaching the edge of the volume (part of a tile); warns if a dropped
    piece is more than a crumb."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(vol < 0, structure=np.ones((3, 3, 3)))
    if n < 2:
        return
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = {int(np.argmax(sizes))}
    for face in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
        keep |= set(np.unique(face).tolist())
    drop = np.array([j for j in range(1, n + 1) if j not in keep])
    if len(drop):
        big = sizes[drop].max() * voxel ** 3
        if big > 0.05:
            say(f"  WARNING: dropping a floating piece of {big:.2f} mm^3")
        vol[np.isin(lab, drop)] = 0.05


def fill_voids(vol):
    """Fill air pockets sealed inside the solid (where texture cuts closed
    over). Air that reaches the edge of the volume is outside air: the base
    cavity reaches it through the open bottom, the shell through its holes."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(vol >= 0)
    if n < 2:
        return
    edge = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                     lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    sealed = np.setdiff1d(np.arange(1, n + 1), edge)
    if len(sealed):
        vol[np.isin(lab, sealed)] = -0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.3, help="mesh resolution, mm")
    ap.add_argument("--preview", action="store_true", help="skip support planning")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    t0 = time.time()
    vol, origin = build(args.voxel)
    mesh = decimate(mesh_volume(vol, args.voxel, origin), 0.006)
    print(f"honu: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, {time.time() - t0:.0f}s")
    out = os.path.join(args.outdir, "honu.stl")
    mesh.export(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
