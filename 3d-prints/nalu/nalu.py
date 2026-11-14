#!/usr/bin/env python3
"""Nalu: a surfer riding a breaking wave (resin / MSLA, Saturn 4 Ultra),
about 165 mm long and 100 mm tall. Two parts, printed separately so the
wave can be clear and the surfer coloured:

  wave      (nalu_wave.stl) a tall wave that peels from left to right. At
            the left it pitches its lip over into a barrel; the barrel end
            is a clean section through the curl. Toward the right the lip
            is just feathering over a steep, concave face, and the wave
            sinks into the sea. The lip's edge is torn into ragged teeth;
            streaks run down the face; foam is pitted along the crest. The
            wave stands on a low oval of rippled sand with a cowrie, a
            cone shell, a plumeria washed up on the sand and a scalloped
            swash line. A shallow bed in the face takes the board, with two
            sockets for its pegs.
  surfer    (nalu_surfer.stl) a surfer in a deep crouch on a longboard,
            riding the face just ahead of the curl: chest to the front,
            front arm reaching down the line, trailing hand held out toward
            the wave behind him, a plumeria lei round his neck. Two keyed
            pegs under the board seat in the wave's sockets.

Printability: the wave is hollow, open to the plate and vented to the back
rim by tunnels; it is solid round the sockets. Its lip hangs over the face
and takes tree supports from the sand. The surfer prints on his own with
tree supports (never on the board's underside, which mates with the wave).

Usage
  python3 nalu.py --voxel 0.4 --preview          # quick look at both parts
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
from lattice_plumeria import BASE_CUT, add, sub  # noqa: E402
from ohia_bonsai import oval  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402
from honu import drop_specks, fill_voids, trim_crumbs  # noqa: E402

# ---------------------------------------------------------------- layout
BASE = (84.0, 50.0)            # half-axes of the base oval
RIM_H = 2.6                    # base height at the rim
HMAX = 95.0                    # height of the barrel's crest
X0, X1 = -72.0, 80.0           # the wave runs from its barrel end (left) to where it sinks (right)
BOARD = dict(x=6.0, L=62.0, W=15.0, T=2.5, zf=0.10)   # zf: height on the face, as a fraction of H
FLOWER = dict(x=46.0, y=-40.0, R=8.5)                  # a plumeria washed up on the sand
COWRIE = (62.0, -36.0, np.radians(-25))
CONE = (-50.0, -42.0, np.radians(15))
WALL = 2.6                     # wall of the hollow wave
TUNNELS = (-52.0, -18.0, 16.0, 50.0)   # x of the vent tunnels under the back of the wave
FIG_HEIGHT = 62.0              # the surfer's standing height, mm
BOARD_TILT = 0.45              # how far the board stands up off the face toward level (0..1)
PEG = dict(r=1.6, flat=0.55, L=3.0, clear=0.15, at=(-14.0, 14.0))   # keyed (D) pegs under the board


def smax(a, b, k):
    return -smin(-a, -b, k)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ell(x, y, z, rx, ry, rz):
    k0 = np.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
    k1 = np.sqrt((x / rx ** 2) ** 2 + (y / ry ** 2) ** 2 + (z / rz ** 2) ** 2)
    return np.where(k1 > 1e-9, k0 * (k0 - 1) / np.maximum(k1, 1e-9), -min(rx, ry, rz))


# ------------------------------------------------------- wave cross-section
# Profiles in (y, z/H): y in mm (the wave breaks toward -y, the viewer), z as
# a fraction of the local height. Listed round the outline.
BARREL = np.array([
    (40, -0.05), (38, 0.0), (32, 0.06), (24, 0.20), (15, 0.42), (8, 0.67), (2.5, 0.86),
    (-4, 0.975), (-12, 1.0), (-20, 0.975), (-27, 0.92), (-32.5, 0.83), (-35.5, 0.71),
    (-36, 0.58), (-34.5, 0.47), (-31.5, 0.40), (-28.5, 0.40), (-27, 0.45), (-28.2, 0.52),
    (-29.2, 0.60), (-28.5, 0.70), (-25, 0.79), (-19, 0.845), (-12, 0.85), (-7, 0.80),
    (-4, 0.68), (-4, 0.50), (-6.5, 0.36), (-11.5, 0.22),
    (-18, 0.105), (-26, 0.035), (-36, 0.0), (-40, -0.05)])
BARREL_LIP = slice(10, 19)
SHOULDER = np.array([
    (40, -0.05), (38, 0.0), (32, 0.07), (24, 0.24), (15, 0.47), (8, 0.70), (2.5, 0.87),
    (-4, 0.97), (-10, 1.0), (-15, 0.985), (-19, 0.95), (-21.5, 0.90), (-22.2, 0.85),
    (-20.8, 0.815), (-18.5, 0.825), (-15, 0.79), (-12, 0.72), (-10, 0.62), (-9.5, 0.50),
    (-10.5, 0.38), (-13, 0.26), (-17.5, 0.15), (-25, 0.06), (-32, 0.02), (-40, -0.05)])
SHOULDER_LIP = slice(9, 15)


def height(x):
    """The full barrel at the left, easing down to a steep shoulder at the
    right, which sinks into the sea at the right end."""
    return HMAX * (1 - 0.42 * sstep(-30, 70, x)) * (1 - 0.8 * sstep(62, X1 + 4, x))


def curl(x):
    """1 where the wave throws a barrel, 0 where the lip only feathers."""
    return 1 - sstep(-24, 42, x)


def bend(x):
    """In plan the barrel end swings toward the viewer."""
    return -8.0 * (1 - (x - X0) / (X1 - X0)) ** 2


def chaikin(p, n=3):
    for _ in range(n):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], np.stack([q, r], 1).reshape(-1, 2), p[-1:]])
    return p


def teeth(x):
    """How far the lip's torn edge reaches at x: ragged teeth, uneven."""
    t = np.abs(np.sin(x / 2.4 + 0.7 * np.sin(x / 7.0))) ** 0.6
    return 1.5 * t + 0.9 * np.sin(x / 9.1) + 0.6 * np.sin(x / 3.7 + 2.0)


def outline(prof, lip, x, H):
    p = prof.astype(float).copy()
    k = 0.5 + 0.5 * H / HMAX                       # a lower wave is also a narrower one
    w = np.zeros(len(p))
    w[lip] = np.sin(np.linspace(0.2, np.pi - 0.2, lip.stop - lip.start))
    p[:, 0] -= w * teeth(x)
    p[:, 1] += w * (0.6 * np.sin(x / 5.2 + 2.0)) / max(H, 1.0)
    p[:, 0] *= k
    p[:, 1] *= H
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


def wave_shape(x, Y, Z):
    """Wave SDF on one x-slice, before texture."""
    H, c = height(x), curl(x)
    Y = Y - bend(x)
    d = None
    if c > 0.01:
        d = poly_sdf(Y, Z, outline(BARREL, BARREL_LIP, x, H))
    if c < 0.99:
        ds = poly_sdf(Y, Z, outline(SHOULDER, SHOULDER_LIP, x, H))
        d = ds if d is None else c * d + (1 - c) * ds
    # the barrel end: the lip throws on past the face
    xend = X0 - 9.0 * sstep(0.5 * H, 0.95 * H, Z) * sstep(-6, -22, Y) - 3.0 * sstep(0.3 * H, 0.9 * H, Z)
    d = smax(d, xend - x, 3.0)
    return smax(d, x - X1, 5.0)                  # the right end rounds off into the sea


def wave_slice(x, Y, Z):
    d = wave_shape(x, Y, Z)
    H = height(x)
    # streaks: fine, uneven lines running down the face and the back
    ph = (x + 2.4 * np.sin(Z / 9.0 + Y / 12.0) + 1.2 * np.sin(Z / 3.7 - x / 9.0) + 0.8 * np.sin(Y / 2.3)) / 3.3
    g = 0.5 + 0.5 * np.cos(2 * np.pi * ph)
    amp = 0.08 + 0.12 * (0.5 + 0.5 * np.sin(x / 6.3 + Z / 17.0))
    face = np.clip(1 - np.abs(d) / 0.8, 0, 1)
    return d + amp * g ** 10 * face * sstep(0.04 * H, 0.12 * H, Z)


def foam_pits(x, y, z, d, n, s=1.5, r=0.5, depth=0.45):
    """Bubbly foam: a jittered scatter of round pits cut into the surface
    along the normal; none where the surface overhangs."""
    q = np.stack([x / s, y / s + 0.5 * np.floor(x / s), z / s + 0.5 * np.floor(y / s)], -1)
    i = np.floor(q)
    h = np.sin(i[..., 0] * 12.9898 + i[..., 1] * 78.233 + i[..., 2] * 37.719) * 43758.5453
    j = h - np.floor(h)
    rr = r * (0.55 + 0.6 * j)
    f = q - i - 0.5
    along = np.einsum("...i,...i->...", f, n)
    cell = np.sqrt(np.maximum(np.einsum("...i,...i->...", f, f) - along ** 2, 0)) * s
    wgt = np.clip((n[..., 2] + 0.2) / 0.3, 0, 1)
    return np.maximum(d, np.minimum(rr - cell, d + depth * wgt))


def foam_zone(x, y, z):
    """Foam along the top of the crest, behind the lip, and the swash line."""
    H = height(x)
    k = 0.5 + 0.5 * H / HMAX
    yw, y = y, (y - bend(x)) / k
    zc = z / np.maximum(H, 1e-3)
    w = np.clip(1 - np.hypot((y + 9) / 9, (zc - 0.99) / 0.06), 0, 1)
    sw = np.exp(-((yw - swash_y(x)) / 2.5) ** 2) * (z < 6)
    return np.maximum(w, sw)


# ------------------------------------------------------------------ base

def swash_y(x):
    return -46.0 + 2.0 * np.sin(x / 9.0) + 1.1 * np.sin(x / 4.1 + 1.3)


def base_top(x, y):
    front = 1 - sstep(-44, -34, y)
    rip = 0.38 * np.sin(0.75 * (x + 2.0 * np.sin(y / 5.0))) * front
    swell = 0.30 * np.sin(0.45 * y + 0.6 * np.sin(x / 13.0)) * sstep(36, 42, y)
    step = 0.45 * sstep(-0.6, 0.6, y - swash_y(x))
    return RIM_H + 1.1 + rip + swell + step


def base_sdf(x, y, z):
    flare = 1.4 * np.clip((RIM_H - z) / RIM_H, 0, 1)
    d = np.maximum(oval(x, y, BASE[0] + flare, BASE[1] + flare), z - base_top(x, y))
    return np.maximum(d, -z)


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
    cx, cy, a = CONE
    c = np.array([cx, cy, base_top(cx, cy) + 0.9])
    ca, sa = np.cos(a), np.sin(a)
    L = 13.0

    def f(x, y, z):
        u = (x - c[0]) * ca + (y - c[1]) * sa + L / 2
        v = -(x - c[0]) * sa + (y - c[1]) * ca
        w = z - c[2]
        t = np.clip(u / L, 0, 1)
        r = 4.4 * np.where(t < 0.82, t / 0.82, 1 - 0.55 * (t - 0.82) / 0.18)
        d = (np.hypot(v, w) - r) * 0.92
        d = np.maximum(d, np.maximum(-u, u - L))
        return d + 0.18 * (0.5 + 0.5 * np.cos(2 * np.pi * u / 1.6)) * (t < 0.3)
    return f, (c - [10, 10, 5], c + [10, 10, 6])


def plumeria(R, lift=None):
    """A five-petal plumeria in its own frame (u, v, w), face up, throat at
    the origin. Petals are obovate, pinwheeled, cupped, tips curling up."""
    lift = 0.26 * R if lift is None else lift

    def petal(u, v, w):
        t = np.clip(u / R, 0, 1)
        wid = 0.47 * R * np.minimum(t / 0.62, 1) ** 0.8 * np.clip(1 - (np.maximum(t - 0.62, 0) / 0.38) ** 2.2, 0, 1)
        wid = wid * np.where(v > 0, 1.1, 0.9) + 1e-3
        outl = np.maximum(np.abs(v) - wid, np.maximum(-u, u - R))
        vn = np.clip(v / wid, -1, 1)
        tip = np.maximum(t - 0.55, 0) / 0.45
        mid = (lift * np.sin(np.pi * 0.5 * np.minimum(t * 1.5, 1)) + 0.07 * R * vn * t
               + 0.06 * R * vn * vn * t + 0.13 * R * tip * tip)
        thick = max(0.11 * R, 0.7) + 0.05 * R * (1 - t)
        return smax(outl, np.abs(w - mid) - thick / 2, 0.025 * R)

    def f(u0, v0, w0):
        d = np.full(u0.shape, 1e9)
        for k in range(5):
            a = 2 * np.pi * k / 5 + 0.3
            off = 0.15 * R
            cx_, cy_ = -off * np.sin(a), off * np.cos(a)
            u = (u0 - cx_) * np.cos(a) + (v0 - cy_) * np.sin(a)
            v = -(u0 - cx_) * np.sin(a) + (v0 - cy_) * np.cos(a)
            d = np.minimum(d, petal(u, v, w0))
        rr = np.hypot(u0, v0)
        dome = np.sqrt(rr ** 2 + (np.maximum(w0 - 0.1 * R, 0) * 1.5) ** 2) - 0.22 * R
        dome = np.maximum(dome, -w0 - 0.25 * R)
        fun = np.sqrt(rr ** 2 + (w0 - 0.1 * R - 0.13 * R) ** 2) - 0.08 * R
        return smin(d, smax(dome, -fun, 0.016 * R), 0.05 * R)
    return f


def flower_sdf():
    x0, y0, R = FLOWER["x"], FLOWER["y"], FLOWER["R"]
    C = np.array([x0, y0, base_top(x0, y0) - 0.3])
    f = plumeria(R)
    return (lambda x, y, z: f(x - C[0], y - C[1], z - C[2])), (C - [R + 3, R + 3, 3], C + [R + 3, R + 3, 6])


# ------------------------------------------------- board on the face

def face_point(x0, z0):
    """The front of the wave at (x0, z0): the first solid met coming from
    the viewer's side, and the outward surface normal there."""
    ys = np.linspace(-60, 40, 2001)
    zz = np.full_like(ys, z0)
    d = wave_shape(x0, ys, zz)
    j = int(np.argmax(d < 0))
    y0 = ys[j - 1] - d[j - 1] * (ys[j] - ys[j - 1]) / (d[j] - d[j - 1])
    e = 0.2

    def s(x, y, z):
        return float(wave_shape(x, np.array([y]), np.array([z]))[0])
    n = np.array([s(x0 + e, y0, z0) - s(x0 - e, y0, z0), s(x0, y0 + e, z0) - s(x0, y0 - e, z0),
                  s(x0, y0, z0 + e) - s(x0, y0, z0 - e)])
    return np.array([x0, y0, z0]), n / np.linalg.norm(n)


def board_frame():
    """The board lies on the face, along the wave: its centre C (on the
    bottom), axis eu (toward the nose, +x), normal en (out of the face) and
    ev = en x eu (toward the wave)."""
    x0, L = BOARD["x"], BOARD["L"]
    z0 = BOARD["zf"] * height(x0)
    pts, ns = zip(*[face_point(x0 + s * 0.3 * L, z0) for s in (-1, 0, 1)])
    pts, ns = np.array(pts), np.array(ns)
    eu = pts[2] - pts[0]
    eu /= np.linalg.norm(eu)
    en = ns.mean(0)
    en -= (en @ eu) * eu
    en /= np.linalg.norm(en)
    # a surfer rides with the board flatter than the face: the uphill rail
    # sets into the face (cut away from the wave), the other stands off it
    up = np.array([0, 0, 1.0]) - eu[2] * eu
    en = (1 - BOARD_TILT) * en + BOARD_TILT * up / np.linalg.norm(up)
    en /= np.linalg.norm(en)
    ev = np.cross(en, eu)
    return pts[1], eu, ev, en


def local(x, y, z, C, eu, ev, en):
    px, py, pz = x - C[0], y - C[1], z - C[2]
    return (px * eu[0] + py * eu[1] + pz * eu[2], px * ev[0] + py * ev[1] + pz * ev[2],
            px * en[0] + py * en[1] + pz * en[2])


def board_sdf(u, v, w):
    """A longboard in its frame: rounded nose (+u), squared-off tail, rocker,
    a lens-shaped section with a soft rail. Its bottom is near w = 0."""
    L, W, T = BOARD["L"], BOARD["W"], BOARD["T"]
    s = u / (L / 2)
    half = 0.5 * W * np.sqrt(np.clip(1 - np.abs(s) ** np.where(s > 0, 2.2, 5.0), 0, 1)) + 1e-3
    rocker = 3.0 * np.clip(s - 0.45, 0, None) ** 2 / 0.55 ** 2 + 1.0 * np.clip(-s - 0.6, 0, None) ** 2 / 0.16
    vn = np.clip(np.abs(v) / half, 0, 1)
    thick = T * np.sqrt(np.clip(1 - vn ** 4, 0.0, 1)) * (1 - 0.35 * np.abs(s) ** 3) + 0.35
    mid = rocker + thick / 2
    d = np.maximum(np.abs(v) - half, np.abs(w - mid) - thick / 2)
    return smax(d, np.abs(u) - L / 2, 0.6)


def peg_sdf(u, v, w, grow=0.0, extra=0.0):
    """D-shaped pegs under the board (the flat stops them turning). grow
    widens them (for the sockets); extra deepens them."""
    r, flat, L = PEG["r"] + grow, PEG["flat"], PEG["L"] + extra
    d = np.full(np.shape(u), 1e9)
    for a in PEG["at"]:
        dd = np.hypot(u - a, v) - r
        dd = np.maximum(dd, v - (r - flat))                      # the flat side faces the wave
        dd = np.maximum(dd, np.maximum(w - 0.5, -w - L))
        d = np.minimum(d, dd)
    return d


# ------------------------------------------------------------- surfer

def figure_frame():
    """Where the posed figure goes: the deck-frame origin (on the deck's
    centre line between the feet) and its axes in the world, and the deck's
    tilt across the board."""
    C, eu, ev, en = board_frame()
    Z = np.array([0, 0, 1.0])
    U = eu - eu[2] * Z
    U /= np.linalg.norm(U)
    Vh = np.cross(Z, U)                                  # level, toward the wave
    tilt = np.degrees(np.arctan2(-(en @ Vh), en @ Z))
    O = C + en * (BOARD["T"] + 0.35 - 0.3) + eu * (-1.0)   # feet sit 0.3 mm into the deck
    return O, U, Vh, Z, tilt


def to_world(p, O, U, Vh, Z):
    p = np.asarray(p, float)
    return O + p[..., :1] * U + p[..., 1:2] * Vh + p[..., 2:3] * Z


def figure():
    """The posed surfer in world coordinates: vertices, triangles, per-vertex
    shorts and hair masks, and the lei's flowers."""
    import surfer
    O, U, Vh, Z, tilt = figure_frame()
    P = surfer.pose(tilt, height=FIG_HEIGHT)
    V = to_world(P["V"], O, U, Vh, Z)
    J = {k: (to_world(a_, O, U, Vh, Z), to_world(b_, O, U, Vh, Z)) for k, (a_, b_) in P["J"].items()}
    # a plumeria lei: a ring of blossoms lying round the base of the neck,
    # lower on the chest than on the back
    nc = J["neck01"][0] - Z * 0.9
    fwd = J["head"][1] - J["head"][0]
    lei = []
    for k in range(12):
        t = 2 * np.pi * k / 12
        hang = 0.5 - 0.5 * np.sin(t)                       # 1 at the front (toward the viewer)
        p = nc + U * 3.9 * np.cos(t) + Vh * 3.4 * np.sin(t) - Z * 2.2 * hang - Vh * 1.2 * hang
        nrm = 0.6 * (U * np.cos(t) + Vh * np.sin(t)) + 0.8 * Z - 0.5 * Vh * hang
        lei.append((p, nrm / np.linalg.norm(nrm), t))
    g = P["garment"]
    tubes = [(to_world(a_, O, U, Vh, Z), to_world(b_, O, U, Vh, Z), ra, rb) for a_, b_, ra, rb in g["tubes"]]
    wp, wn = g["waist"]
    wn = wn[0] * U + wn[1] * Vh + wn[2] * Z
    return dict(V=V, faces=P["faces"], shorts=P["shorts"], hair=P["hair"], lei=lei, J=J,
                garment=dict(tubes=tubes, waist=(to_world(wp, O, U, Vh, Z), wn)))


def shorts_sdf(x, y, z, g):
    """Loose board shorts: tapered tubes with flat ends, blended, cut flat at
    the waist; the thigh tubes' ends are the hems."""
    d = np.full(x.shape, 1e9)
    for k, (a, b, ra, rb) in enumerate(g["tubes"]):
        ab = b - a
        L = np.linalg.norm(ab)
        ax = ab / L
        px, py, pz = x - a[0], y - a[1], z - a[2]
        t = px * ax[0] + py * ax[1] + pz * ax[2]
        r = ra + (rb - ra) * np.clip(t / L, 0, 1)
        radial = np.sqrt(np.maximum(px ** 2 + py ** 2 + pz ** 2 - t ** 2, 0)) - r
        if k < 2:                                            # thigh: flat hem, open to the hip
            dd = np.maximum(np.maximum(radial, t - L), -t - 0.5)
        else:                                                # pelvis: rounded ends
            tc = np.clip(t, 0, L)
            dd = np.sqrt(np.maximum(px ** 2 + py ** 2 + pz ** 2 - 2 * t * tc + tc ** 2, 0)) - r
        d = smin(d, dd, 1.2)
    wp, wn = g["waist"]
    return np.maximum(d, (x - wp[0]) * wn[0] + (y - wp[1]) * wn[1] + (z - wp[2]) * wn[2])


def lei_sdf(x, y, z, lei):
    f = plumeria(1.45, lift=0.5)
    d = np.full(x.shape, 1e9)
    for p, n, t in lei:
        a = np.cross(n, [0.3, 0.2, 0.9])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        px, py, pz = x - p[0], y - p[1], z - p[2]
        u = px * a[0] + py * a[1] + pz * a[2]
        v = px * b[0] + py * b[1] + pz * b[2]
        w = px * n[0] + py * n[1] + pz * n[2]
        cs, sn = np.cos(t * 1.7), np.sin(t * 1.7)
        d = np.minimum(d, f(u * cs - v * sn, u * sn + v * cs, w + 0.3))
    return d


def body_sdf(G, X, Y, Z, spacing=0.05):
    """Signed distance to the posed body, with board shorts and hair laid
    over it. Distances are to a dense sampling of the surface; the sign
    comes from the nearest sample's normal."""
    if "_tree" not in G:
        from scipy.spatial import cKDTree
        import trimesh
        m = trimesh.Trimesh(G["V"], G["faces"], process=False)
        pts, fi = trimesh.sample.sample_surface(m, int(m.area / spacing ** 2), seed=3)
        fv = G["faces"][fi]
        bary = trimesh.triangles.points_to_barycentric(m.vertices[fv], pts)

        def interp(a):
            return np.einsum("ij,ij->i", bary, a[fv])
        G["_tree"] = (cKDTree(pts), pts, m.face_normals[fi], interp(G["shorts"]), interp(G["hair"]))
    tree, pts, nrm, sh, hr = G["_tree"]
    q = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
    dist, j = tree.query(q, workers=-1, distance_upper_bound=4.0)
    far = ~np.isfinite(dist)
    j[far] = 0
    side = np.einsum("ij,ij->i", q - pts[j], nrm[j])
    d = np.where(side < 0, -dist, dist)
    # shorts stand 0.35 mm proud of the skin, with a crisp hem and a waistband
    # hair: 0.6 mm over the scalp, combed back in fine grooves
    hm = hr[j]
    head = G["J"]["head"]
    ax = head[1] - head[0]
    ax /= np.linalg.norm(ax)
    rel = q - head[0]
    side_ax = np.cross(ax, [0, 0, 1.0]) + 1e-9
    side_ax /= np.linalg.norm(side_ax)
    g = 0.5 + 0.5 * np.cos(2 * np.pi * (rel @ side_ax + 0.4 * np.sin(rel @ ax * 0.9)) / 0.7)
    d = d - hm * (1.0 - 0.3 * g ** 4)                    # combed-back waves, thin at the hairline
    d[far] = 4.0
    return d.reshape(X.shape).astype(np.float32)


def surfer_box():
    G = figure()
    C, eu, ev, en = board_frame()
    L = BOARD["L"] / 2 + 4
    lo = np.minimum(G["V"].min(0) - 2, C - L)
    hi = np.maximum(G["V"].max(0) + 2, C + L)
    return lo, hi


# ------------------------------------------------------------------ build

def bounds(part="wave"):
    if part == "surfer":
        return surfer_box()
    return np.array([-BASE[0] - 3, -BASE[1] - 3, -0.6]), np.array([BASE[0] + 3, BASE[1] + 3, HMAX + 6])


def build_surfer(voxel, quiet=False, region=None):
    say = (lambda *a: None) if quiet else print
    t0 = time.time()
    G = figure()
    C, eu, ev, en = board_frame()
    lo, hi = surfer_box() if region is None else (np.asarray(region[0]), np.asarray(region[1]))
    xs = np.arange(lo[0], hi[0], voxel, dtype=np.float32)
    ys = np.arange(lo[1], hi[1], voxel, dtype=np.float32)
    zs = np.arange(lo[2], hi[2], voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    for i in range(0, len(xs), 24):
        sx = slice(i, min(i + 24, len(xs)))
        X, Y, Z = np.meshgrid(xs[sx], ys, zs, indexing="ij")
        u, v, w = local(X, Y, Z, C, eu, ev, en)
        board = np.minimum(board_sdf(u, v, w), peg_sdf(u, v, w))
        body = body_sdf(G, X, Y, Z)
        body = smin(body, shorts_sdf(X, Y, Z, G["garment"]), 0.3)
        body = np.minimum(body, lei_sdf(X, Y, Z, G["lei"]))
        vol[sx] = smin(board, body, 0.4)
    say(f"  surfer {time.time() - t0:.0f}s")
    fill_voids(vol)
    drop_specks(vol, voxel, say)
    return vol, (xs[0], ys[0], zs[0])


def build(voxel, stems=(), quiet=False, region=None):
    """The wave part."""
    say = (lambda *a: None) if quiet else print
    t0 = time.time()
    lo, hi = bounds("wave") if region is None else (np.asarray(region[0]), np.asarray(region[1]))
    xs = np.arange(lo[0], hi[0], voxel, dtype=np.float32)
    ys = np.arange(lo[1], hi[1], voxel, dtype=np.float32)
    zs = np.arange(lo[2], hi[2], voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    add(vol, xs, ys, zs, (-BASE[0] - 3, -BASE[1] - 3, -1), (BASE[0] + 3, BASE[1] + 3, RIM_H + 4), base_sdf, k=0, chunk=8)
    say(f"  base {time.time() - t0:.0f}s")

    C, eu, ev, en = board_frame()
    kb = int(np.searchsorted(zs, 6.0))
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    for i, x in enumerate(xs):
        if x < X0 - 14 or x > X1 + 1:
            continue
        d = wave_slice(float(x), Y, Z).astype(np.float32)
        vol[i] = smin(vol[i], d, 2.0)
        # hollow, WALL in from the skin, carried down through the base so it
        # opens on the plate; vented to the back rim by pitched tunnels; kept
        # solid round the sockets
        cav = d + WALL
        cav[:, Z[0] < 6.0] = cav[:, kb:kb + 1]
        for xt in TUNNELS:
            if abs(x - xt) < 4:
                cav = np.minimum(cav, np.maximum(Z + abs(x - xt) - 2.6, -(Y - bend(x) + 4.0)))
        for a in PEG["at"]:
            pc = C + eu * a
            cav = np.maximum(cav, 7.0 - np.sqrt((x - pc[0]) ** 2 + (Y - pc[1]) ** 2 + (Z - pc[2]) ** 2))
        vol[i] = np.maximum(vol[i], -cav)
    say(f"  wave {time.time() - t0:.0f}s")

    for f, (a, b) in (cowrie_sdf(), cone_sdf(), flower_sdf()):
        add(vol, xs, ys, zs, a, b, f, k=0.6, chunk=16)
    say(f"  shells and flower {time.time() - t0:.0f}s")

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

    # the board's bed and its sockets: the board's shape is cut out of the
    # face, and the pegs' holes PEG clear wider and 0.3 mm deeper
    def cut(x, y, z):
        u, v, w = local(x, y, z, C, eu, ev, en)
        return np.minimum(board_sdf(u, v, w) - 0.05, peg_sdf(u, v, w, grow=PEG["clear"], extra=0.3))
    s_ = sub(xs, ys, zs, C - BOARD["L"] / 2 - 6, C + BOARD["L"] / 2 + 6)
    for i in range(s_[0].start, s_[0].stop, 8):
        sx = slice(i, min(i + 8, s_[0].stop))
        X3, Y3, Z3 = np.meshgrid(xs[sx], ys[s_[1]], zs[s_[2]], indexing="ij")
        vol[sx, s_[1], s_[2]] = np.maximum(vol[sx, s_[1], s_[2]], -cut(X3, Y3, Z3))
    say(f"  board bed and sockets {time.time() - t0:.0f}s")

    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])
    fill_voids(vol)
    drop_specks(vol, voxel, say)
    n = trim_crumbs(vol, voxel, zs)
    say(f"  sealed voids filled, specks dropped, {n} crumbs trimmed {time.time() - t0:.0f}s")
    return vol, (xs[0], ys[0], zs[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.4, help="mesh resolution, mm")
    ap.add_argument("--part", choices=("wave", "surfer", "both"), default="both")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    for part in (("wave", "surfer") if args.part == "both" else (args.part,)):
        t0 = time.time()
        v = args.voxel if part == "wave" else (min(args.voxel * 0.5, 0.25) if args.preview else 0.15)
        vol, origin = build(v) if part == "wave" else build_surfer(v)
        mesh = mesh_volume(vol, v, origin)
        if not args.preview:
            mesh = decimate(mesh, 0.006)
        print(f"{part}: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
              f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, {time.time() - t0:.0f}s")
        out = os.path.join(args.outdir, f"nalu_{part}.stl")
        mesh.export(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
