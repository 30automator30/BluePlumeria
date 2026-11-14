#!/usr/bin/env python3
"""Nalu: a surfer riding a breaking wave (resin / MSLA, Saturn 4 Ultra),
about 130 mm long. One part, printed straight on the plate:

  base      a low oval of rippled sand in front, swell lines behind, with
            a cowrie (leho) and a cone shell pressed into the sand and a
            scalloped swash line where the last wash stopped.
  wave      a long, low wave, rounded at both ends, whose crest rolls
            forward in a thick foam lip; below it two more curls break
            down the face, one under the other, and the lowest spills out
            onto the sand. Every curl ripples along its length.
  surfer    a figure in a low crouch on a long board, which lies along the
            crest of the wave; arms out for balance, eyes down the line.
  plumeria  one five-petal blossom riding the crest beyond the board's nose.

Printability: the base and the wave rise from the plate. The underside of
every curl slopes back to the face at 45 degrees, so the curls need no
supports. The board lies along the crest. The surfer's arms, chin and the
board's lifted nose and tail take light tree supports in CHITUBOX. The wave
is hollow, open to the plate, and vented to the back rim by tunnels.

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
from fractal_plumeria import capsule_chain, mesh_volume  # noqa: E402
from lattice_plumeria import BASE_CUT, add  # noqa: E402
from ohia_bonsai import oval  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402
from honu import drop_specks, fill_voids, trim_crumbs  # noqa: E402

# ---------------------------------------------------------------- layout
BASE = (68.0, 42.0)            # half-axes of the base oval
RIM_H = 2.6                    # base height at the rim
HMAX = 50.0                    # height of the crest above the plate
XL = 63.0                      # the wave runs from -XL to XL
BOARD = dict(x=-6.0, L=68.0, W=14.5, T=2.4)
FLOWER = dict(x=44.0, R=12.5, tilt=np.radians(32))
COWRIE = (40.0, -35.0, np.radians(-25))
CONE = (-36.0, -36.5, np.radians(15))
WALL = 2.6                     # wall of the hollow wave
TUNNELS = (-38.0, -6.0, 26.0)  # x of the vent tunnels under the back of the wave


def smax(a, b, k):
    return -smin(-a, -b, k)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------- wave cross-section
# The outline in (y, z), mm, at full height; the wave breaks toward -y (the
# viewer). Groups of points make the crest lip and the two curls on the face.
PROFILE = [
    (34, -3), (32, 0), (28, 6), (23, 16), (18, 27), (12, 37), (6, 44.5), (0, 49), (-6, 50.5),
    # crest lip, rolling forward
    (-12, 50), (-17, 47.5), (-20.5, 43.5), (-22, 39.5), (-21.5, 36), (-19.5, 34),
    (-14.5, 29), (-13, 28.2),                       # underside back to the face, then the face
    # first curl on the face
    (-15, 27), (-18, 24.5), (-19.5, 21.5), (-19, 19), (-17.5, 17.5),
    (-12.5, 12.5), (-11.5, 11.6),
    # second curl, spilling onto the sand
    (-14, 10), (-17, 8), (-19, 5.5), (-20.5, 3.2), (-22, 1.6),
    (-26, 0.5), (-30, 0), (-34, -3)]
GROUPS = [  # (points, attach point under it, y offset fn, z offset fn)
    (range(9, 15), 15, lambda x: 1.2 + 1.6 * np.sin(x / 6.1) + 1.0 * np.sin(x / 2.7 + 1.0),
     lambda x: 0.8 * np.sin(x / 4.3)),
    (range(17, 22), 22, lambda x: 0.6 + 2.0 * np.sin(x / 7.7 + 2.0) + 0.9 * np.sin(x / 3.1),
     lambda x: 1.1 * np.sin(x / 5.9 + 1.0)),
    (range(24, 29), None, lambda x: 0.5 + 1.8 * np.sin(x / 9.3 + 4.0) + 0.8 * np.sin(x / 2.9 + 3.0),
     lambda x: 0.0 * x)]


def envelope(x):
    """0..1: the wave is full height along most of its length and rolls
    down round both ends."""
    return np.clip(1 - (np.abs(x) / XL) ** 2.5, 0, 1) ** 0.45


def bend(x):
    """A gentle crescent in plan: both ends come a little toward the viewer."""
    return 3.0 - 6.0 * (x / XL) ** 2


def chaikin(p, n=3):
    for _ in range(n):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], np.stack([q, r], 1).reshape(-1, 2), p[-1:]])
    return p


def outline(x):
    """The outline for slice x: each curl ripples in and out along the wave,
    and the point where its underside meets the face moves with it, so the
    underside stays at 45 degrees."""
    p = np.array(PROFILE, float)
    for idx, attach, oy, oz in GROUPS:
        idx = list(idx)
        w = np.sin(np.linspace(0.35, np.pi - 0.35, len(idx)))
        w = w / w.max()
        p[idx, 0] -= w * oy(x)
        p[idx, 1] += w * oz(x)
        if attach is not None:
            by, bz = p[idx[-1]]
            fy = p[attach + 1, 0]                       # the face just below
            p[attach] = (fy - 1.5, bz - (fy - 1.5 - by))  # 45 degrees from the curl's foot
    e = envelope(x)
    p[:, 1] *= e
    p[:, 0] *= 0.55 + 0.45 * e
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


def wave_slice(x, Y, Z):
    """Wave SDF on one x-slice."""
    Y = Y - bend(x)
    d = poly_sdf(Y, Z, outline(x))
    H = HMAX * envelope(x)
    # cut-in flow lines, uneven in spacing and depth, streaming down the
    # back and the face
    ph = (x + 2.6 * np.sin(Z / 9.0 + Y / 12.0) + 1.4 * np.sin(Z / 3.7 - x / 9.0)) / 4.6
    g = 0.5 + 0.5 * np.cos(2 * np.pi * ph)
    amp = 0.22 + 0.16 * (0.5 + 0.5 * np.sin(x / 6.3 + Z / 17.0))
    face = np.clip(1 - np.abs(d) / 0.8, 0, 1)
    d = d + amp * g ** 8 * face * sstep(0.04 * H, 0.12 * H, Z)
    # a curl line cut into the front of each roll, where the lip folds over
    for zc, yc in ((0.80, -21.0), (0.42, -19.0)):
        k = 0.55 + 0.45 * envelope(x)
        r = np.hypot((Y / k - yc) / 3.2, (Z - zc * H) / 2.4)
        d = d + 0.45 * np.exp(-((r - 1.0) / 0.18) ** 2) * (Y / k < yc + 0.5) * face
    return d


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
    """Where the wave is foamy, 0..1: the top of the crest lip and of each
    curl, and the swash line on the sand."""
    H = HMAX * envelope(x)
    k = 0.55 + 0.45 * envelope(x)
    yw, y = y, (y - bend(x)) / k
    zc = z / np.maximum(H, 1e-3)
    w = np.clip(1 - np.hypot((y + 12) / 10, (zc - 0.97) / 0.09), 0, 1)
    w = np.maximum(w, np.clip(1 - np.hypot((y + 16) / 5, (zc - 0.52) / 0.07), 0, 1))
    w = np.maximum(w, np.clip(1 - np.hypot((y + 17) / 6, (zc - 0.19) / 0.08), 0, 1))
    sw = np.exp(-((yw - swash_y(x)) / 2.5) ** 2) * (z < 6)
    return np.maximum(w, sw)


# ------------------------------------------------------------------ base

def swash_y(x):
    return -38.5 + 1.8 * np.sin(x / 9.0) + 1.0 * np.sin(x / 4.1 + 1.3)


def base_top(x, y):
    """Sand in front (ripples), swell lines behind the wave."""
    front = 1 - sstep(-34, -26, y)
    rip = 0.38 * np.sin(0.75 * (x + 2.0 * np.sin(y / 5.0))) * front
    swell = 0.30 * np.sin(0.45 * y + 0.6 * np.sin(x / 13.0)) * sstep(28, 34, y)
    step = 0.45 * sstep(-0.6, 0.6, y - swash_y(x))     # scalloped edge of the last wash
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
    """A cone shell lying on its side, half in the sand."""
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
        d = (np.hypot(v, w) - r) * 0.92
        d = np.maximum(d, np.maximum(-u, u - L))
        return d + 0.18 * (0.5 + 0.5 * np.cos(2 * np.pi * u / 1.6)) * (t < 0.3)   # whorls on the spire
    return f, (c - [10, 10, 5], c + [10, 10, 6])


# ------------------------------------------------- crest, board and surfer

def crest(x0):
    """(y, z) of the top of the wave in slice x0."""
    ys = np.linspace(-40, 40, 401)
    zs = np.linspace(0, HMAX + 5, 551)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    solid = wave_slice(x0, Y, Z) < 0
    top = np.where(solid.any(1), len(zs) - 1 - np.argmax(solid[:, ::-1], 1), 0)
    j = int(np.argmax(top))
    return ys[j], zs[top[j]]


def board_frame():
    """The board's centre on the crest and its tilt: it lies along the crest
    line, following the crest's slope between tail and nose."""
    x0, L = BOARD["x"], BOARD["L"]
    yt, zt = crest(x0 - 0.3 * L)
    yn, zn = crest(x0 + 0.3 * L)
    yc, zc = crest(x0)
    return np.array([x0, yc, zc]), np.arctan2(zn - zt, 0.6 * L), np.arctan2(yn - yt, 0.6 * L)


def board_sdf(x, y, z, C, pitch, yaw):
    """A longboard: rounded nose (+u), squared-off tail, rocker at both ends,
    a lens-shaped section with a soft rail."""
    L, W, T = BOARD["L"], BOARD["W"], BOARD["T"]
    px, py = x - C[0], y - C[1]
    u = px * np.cos(yaw) + py * np.sin(yaw)
    v = -px * np.sin(yaw) + py * np.cos(yaw)
    w = z - C[2] - u * np.tan(pitch)
    s = u / (L / 2)                                     # -1 tail .. 1 nose
    half = 0.5 * W * np.sqrt(np.clip(1 - np.abs(s) ** np.where(s > 0, 2.2, 5.0), 0, 1)) + 1e-3
    rocker = 3.4 * np.clip(s - 0.35, 0, None) ** 2 / 0.65 ** 2 + 1.2 * np.clip(-s - 0.5, 0, None) ** 2 / 0.25
    bottom = -0.7 + rocker                               # the board sits into the foam
    vn = np.clip(np.abs(v) / half, 0, 1)
    thick = T * np.sqrt(np.clip(1 - vn ** 2, 0.0, 1)) * (1 - 0.35 * np.abs(s) ** 3) + 0.4
    mid = bottom + thick / 2 - 0.4
    d = np.maximum(np.abs(v) - half, np.abs(w - mid) - thick / 2)
    return smax(d, np.abs(u) - L / 2, 0.6)


def surfer_parts(C, pitch, yaw):
    """Capsule chains (points, radii) and ellipsoids for the surfer, in a
    frame on the board's deck: u along the board (+ toward the nose), v
    across it (- toward the viewer), w up. A low, wide crouch, front knee
    over the front foot, back knee tucked; torso turned to the viewer, eyes
    down the line toward the nose; arms out for balance."""
    T = BOARD["T"]
    deck = C[2] - 0.7 + T * 0.95

    def P(u, v, w):
        return np.array([C[0] + u * np.cos(yaw) - v * np.sin(yaw),
                         C[1] + u * np.sin(yaw) + v * np.cos(yaw),
                         deck + w + u * np.tan(pitch)])
    chains = [
        # legs: ankle, knee, hip (a deep crouch; the front knee over the toes)
        ([P(-15, 0.5, 1.6), P(-13.0, -1.8, 6.5), P(-11.0, -3.0, 10.5), P(-6.0, -0.5, 18.5)], [1.3, 1.9, 2.25, 3.2]),
        ([P(7.0, 0.0, 1.6), P(8.2, -1.6, 6.8), P(9.0, -3.0, 11.0), P(0.0, -0.5, 19.0)], [1.3, 1.9, 2.3, 3.2]),
        # torso: pelvis, waist, chest, neck, leaning toward the nose
        ([P(-3.0, -0.5, 19.0), P(-1.8, -1.3, 24.0), P(0.0, -2.0, 29.0), P(1.2, -2.4, 33.2)],
         [3.8, 3.2, 3.9, 1.8]),
        # shoulders: a bar across the top of the chest
        ([P(-4.8, -1.4, 31.2), P(0.6, -2.2, 31.8), P(5.2, -2.7, 31.2)], [2.0, 2.4, 2.0]),
        # back arm raised behind: shoulder, elbow, wrist
        ([P(-4.6, -1.6, 31.0), P(-10.0, -1.0, 34.5), P(-15.0, 0.0, 38.5)], [1.65, 1.35, 1.0]),
        # front arm reaching down the line, low: shoulder, elbow, wrist
        ([P(5.0, -2.6, 31.0), P(10.5, -5.0, 28.5), P(15.5, -7.0, 25.5)], [1.65, 1.35, 1.0]),
    ]
    blobs = [  # centre, radii (world-axis ellipsoids)
        (P(-15.6, 0.6, 0.9), (3.6, 1.6, 1.0)),          # back foot
        (P(7.6, 0.0, 0.9), (3.6, 1.6, 1.0)),            # front foot
        (P(0.2, -2.1, 28.8), (5.2, 3.0, 3.6)),          # chest, broad across the shoulders
        (P(-3.0, -0.6, 19.2), (5.0, 3.2, 3.0)),         # hips and board shorts
        (P(2.2, -3.0, 36.8), (2.7, 2.9, 3.2)),          # head
        (P(1.6, -2.8, 38.1), (2.8, 2.9, 2.3)),          # hair
        (P(3.2, -3.4, 35.0), (1.8, 1.8, 1.6)),          # jaw and chin
        (P(4.9, -3.6, 36.6), (0.8, 0.6, 0.9)),          # nose: he looks down the line
        (P(-16.2, 0.2, 39.4), (1.0, 1.4, 1.5)),         # hands
        (P(16.6, -7.4, 24.7), (1.5, 1.0, 1.2)),
    ]
    return chains, blobs


def rider_sdf():
    C, pitch, yaw = board_frame()
    chains, blobs = surfer_parts(C, pitch, yaw)

    def f(x, y, z):
        d = board_sdf(x, y, z, C, pitch, yaw)
        s = np.full(x.shape, 1e9)
        for pts, rr in chains:
            s = smin(s, capsule_chain(x, y, z, np.array(pts), np.array(rr)), 0.9)
        for c, r in blobs:
            s = smin(s, ell(x - c[0], y - c[1], z - c[2], *r), 0.8)
        return smin(d, s, 0.5)
    lo = C - [BOARD["L"] / 2 + 3, BOARD["W"] + 6, 3]
    hi = C + [BOARD["L"] / 2 + 3, BOARD["W"] + 6, 50]
    return f, (lo, hi)


# -------------------------------------------------------------- plumeria

def plumeria_sdf():
    """A five-petal plumeria resting on the crest. Petals are obovate,
    twisted pinwheel-fashion, cupped and curling up at the tip; they all rise
    from the throat, so none hangs below it."""
    x0, R, tilt = FLOWER["x"], FLOWER["R"], FLOWER["tilt"]
    yc, zc = crest(x0)
    C = np.array([x0, yc + 1.5, zc - 0.4])
    ct, st = np.cos(tilt), np.sin(tilt)
    Rm = np.array([[1, 0, 0], [0, ct, st], [0, -st, ct]])    # tip it toward the viewer

    def petal(u, v, w):
        t = np.clip(u / R, 0, 1)
        wid = 0.47 * R * np.minimum(t / 0.62, 1) ** 0.8 * np.clip(1 - (np.maximum(t - 0.62, 0) / 0.38) ** 2.2, 0, 1)
        wid = wid * np.where(v > 0, 1.1, 0.9) + 1e-3
        outl = np.maximum(np.abs(v) - wid, np.maximum(-u, u - R))
        vn = np.clip(v / wid, -1, 1)
        tip = np.maximum(t - 0.55, 0) / 0.45
        mid = (3.2 * np.sin(np.pi * 0.5 * np.minimum(t * 1.5, 1)) + 0.9 * vn * t
               + 0.8 * vn * vn * t + 1.6 * tip * tip)
        thick = 1.35 + 0.6 * (1 - t)
        return smax(outl, np.abs(w - mid) - thick / 2, 0.3)

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
        return smin(d, smax(dome, -fun, 0.2), 0.6)
    return f, (C - [R + 7, R + 7, 9], C + [R + 7, R + 7, 15])


# ------------------------------------------------------------------ build

def bounds():
    return np.array([-BASE[0] - 3, -BASE[1] - 3, -0.6]), np.array([BASE[0] + 3, BASE[1] + 3, HMAX + 52])


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

    # the wave, slice by slice, blended into the base, then hollowed: its
    # inside, WALL in from the skin, carried straight down through the base
    # so it opens on the plate, vented to the back rim by tunnels with
    # 45-degree pitched roofs
    kz = zs < HMAX + 4
    kb = int(np.searchsorted(zs, 6.0))
    Y, Z = np.meshgrid(ys, zs[kz], indexing="ij")
    for i, x in enumerate(xs):
        if abs(x) > XL + 1:
            continue
        d = wave_slice(float(x), Y, Z).astype(np.float32)
        vol[i][:, kz] = smin(vol[i][:, kz], d, 2.0)
        cav = d + WALL
        cav[:, Z[0] < 6.0] = cav[:, kb:kb + 1]
        for xt in TUNNELS:
            if abs(x - xt) < 4:
                cav = np.minimum(cav, np.maximum(Z + abs(x - xt) - 2.6, -(Y - bend(x) + 4.0)))
        vol[i][:, kz] = np.maximum(vol[i][:, kz], -cav)
    say(f"  wave {time.time() - t0:.0f}s")

    for f, (a, b) in (cowrie_sdf(), cone_sdf()):
        add(vol, xs, ys, zs, a, b, f, k=0.6, chunk=16)
    say(f"  shells {time.time() - t0:.0f}s")

    # foam: pits cut along the crest, the tops of the curls and the swash line
    idx = np.argwhere(np.abs(vol) < 1.0)
    idx = idx[np.all((idx > 0) & (idx < np.array(vol.shape) - 1), axis=1)]
    for c0 in range(0, len(idx), 4_000_000):
        ii = idx[c0:c0 + 4_000_000]
        a_, b_, c_ = ii[:, 0], ii[:, 1], ii[:, 2]
        px, py, pz = xs[a_], ys[b_], zs[c_]
        m = foam_zone(px, py, pz) > 0.25
        if not m.any():
            continue
        a_, b_, c_, px, py, pz = a_[m], b_[m], c_[m], px[m], py[m], pz[m]
        g = np.stack([vol[a_ + 1, b_, c_] - vol[a_ - 1, b_, c_], vol[a_, b_ + 1, c_] - vol[a_, b_ - 1, c_],
                      vol[a_, b_, c_ + 1] - vol[a_, b_, c_ - 1]], -1)
        g /= np.linalg.norm(g, axis=1)[:, None] + 1e-9
        vol[a_, b_, c_] = foam_pits(px, py, pz, vol[a_, b_, c_], g)
    say(f"  foam {time.time() - t0:.0f}s")

    for f, (a, b) in (rider_sdf(), plumeria_sdf()):
        add(vol, xs, ys, zs, a, b, f, k=0.8, chunk=8)
    say(f"  surfer and plumeria {time.time() - t0:.0f}s")
    for pts, rr in stems:
        add(vol, xs, ys, zs, pts.min(0) - 2.5, pts.max(0) + 2.5,
            lambda x, y, z, s_=pts, r_=rr: capsule_chain(x, y, z, s_, r_), k=0.8)
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
