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
SURFER_SCALE = 1.15             # the figure against the board and the wave
BULK = 1.1                     # thicker limbs and torso: a strong, statue-like build
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
    (-4, 0.975), (-11, 1.0), (-18, 0.965), (-23.5, 0.89), (-27.5, 0.78), (-29, 0.67),
    (-28, 0.585), (-26, 0.545), (-24.6, 0.575), (-25.2, 0.65), (-23.5, 0.74), (-18, 0.815),
    (-11, 0.82), (-6.5, 0.76), (-4, 0.64), (-4, 0.50), (-6.5, 0.36), (-11.5, 0.22),
    (-18, 0.105), (-26, 0.035), (-36, 0.0), (-40, -0.05)])
BARREL_LIP = slice(10, 17)
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
    return 1 - sstep(-34, 30, x)


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

def ik(S, T, L1, L2, pole):
    """Elbow or knee for a two-bone limb from S reaching toward T, bending
    toward pole. Returns (joint, end), the end pulled in if out of reach."""
    D = T - S
    dl = min(np.linalg.norm(D), (L1 + L2) * 0.999)
    Dn = D / np.linalg.norm(D)
    T = S + Dn * dl
    a = (L1 ** 2 - L2 ** 2 + dl ** 2) / (2 * dl)
    h = np.sqrt(max(L1 ** 2 - a ** 2, 0))
    p = pole - (pole @ Dn) * Dn
    p /= np.linalg.norm(p)
    return S + Dn * a + p * h, T


def surfer_parts():
    """Capsule chains and oriented ellipsoids for the surfer, in world
    coordinates. Body frame: U along the board toward the nose (level), V
    toward the wave, Z up; origin on the deck between the feet."""
    C, eu, ev, en = board_frame()
    T = BOARD["T"]
    Z = np.array([0, 0, 1.0])
    U = eu - eu[2] * Z
    U /= np.linalg.norm(U)
    V = np.cross(Z, U)
    deck0 = C + en * (T + 0.35)

    def deck(u):
        return deck0 + eu * u

    O = deck(-1.0)

    def P(u, v, w):
        return O + U * u + V * v + Z * w

    BODY = np.stack([U, V, Z], 1)
    BOARDR = np.stack([eu, ev, en], 1)
    chains, ells = [], []

    def E(c, r, R=BODY):
        ells.append((np.asarray(c, float), R, r))

    # feet across the board, toes toward the viewer, flat on the tilted deck
    fb, ff = deck(-13.0) + en * 0.8, deck(11.0) + en * 0.8
    E(fb - ev * 0.8, (1.35, 3.8, 0.95), BOARDR)
    E(ff - ev * 0.8, (1.35, 3.8, 0.95), BOARDR)
    ankle_b, ankle_f = fb + ev * 1.6 + en * 1.3, ff + ev * 1.6 + en * 1.3
    # pelvis and hips
    pel = P(-0.8, 0.9, 18.0)
    hip_b, hip_f = P(-3.6, 0.9, 17.8), P(2.0, 0.6, 18.2)
    # legs, thigh 12.3 and shin 12.0: a deep crouch, the front knee over the
    # toes, the back knee dropped low and in
    knee_f, _ = ik(hip_f, ankle_f, 12.3, 12.0, -V * 0.8 + U * 0.6)
    knee_b, _ = ik(hip_b, ankle_b, 12.3, 12.0, -V * 0.9 + U * 0.5 - Z * 0.3)
    for hip, knee, ank in ((hip_b, knee_b, ankle_b), (hip_f, knee_f, ankle_f)):
        th, sh = knee - hip, ank - knee
        chains.append(([hip, hip + 0.45 * th, knee], [2.7, 2.35, 1.75]))    # thigh
        chains.append(([knee, knee + 0.3 * sh, ank], [1.9, 2.05, 1.15]))    # shin
        E(knee + 0.12 * th, (1.9, 1.9, 1.9))                                 # kneecap
        E(knee + 0.3 * sh + V * 0.7, (1.6, 1.7, 2.6))                        # calf
    # board shorts over the thighs to just above the knee
    shorts = [([hip, hip + 0.68 * (knee - hip)], [3.05, 2.65]) for hip, knee in ((hip_b, knee_b), (hip_f, knee_f))]
    shorts.append(([P(-3.0, 0.9, 18.5), P(1.6, 0.6, 18.8)], [3.2, 3.2]))
    # torso, leaning toward the nose and a little toward the viewer
    waist, chest, neck0 = P(0.6, 0.3, 23.0), P(2.6, -0.5, 28.6), P(4.5, -1.2, 33.2)
    chains.append(([pel, waist, chest, neck0], [3.5, 3.4, 4.1, 1.95]))
    E(chest + V * 0.3, (5.9, 3.1, 3.9))                                     # ribcage
    E(P(-1.4, 0.9, 26.8), (2.0, 2.3, 3.6))                                   # lats
    E(P(6.6, -0.4, 26.4), (2.0, 2.3, 3.6))
    E(P(0.4, -1.9, 29.6), (2.6, 0.8, 1.6))                                   # pecs, broad and flat
    E(P(4.9, -2.3, 29.2), (2.6, 0.8, 1.6))
    E(P(-1.1, 2.0, 18.6), (3.0, 1.5, 2.0))                                   # seat
    for k in range(3):                                                       # abdominals
        for s in (-0.9, 0.9):
            E(P(1.9 + 0.35 * k + s, -2.2 + 0.25 * k, 25.6 - 2.0 * k), (0.9, 0.55, 0.85))
    # shoulders and arms
    sh_b, sh_f = P(-3.0, 0.2, 31.3), P(8.8, -1.5, 30.4)
    chains.append(([sh_b, P(2.8, -0.6, 32.0), sh_f], [2.2, 2.7, 2.2]))
    E(sh_b + Z * 0.2, (2.5, 2.2, 2.5))                                       # deltoids
    E(sh_f + Z * 0.2, (2.5, 2.2, 2.5))
    E(P(3.2, 1.6, 31.0), (4.6, 2.0, 2.6))                                    # upper back
    # front arm reaching down the line, low; trailing arm out behind toward the wave
    elb_f, wr_f = ik(sh_f, P(20.0, -6.5, 20.5), 7.8, 7.0, -Z * 0.5 - V * 0.5 + U * 0.2)
    elb_b, wr_b = ik(sh_b, P(-12.0, 9.5, 27.5), 7.8, 7.0, -Z * 0.7 - U * 0.4)
    for sh, el, wr in ((sh_f, elb_f, wr_f), (sh_b, elb_b, wr_b)):
        ua, fa = el - sh, wr - el
        chains.append(([sh, sh + 0.45 * ua, el], [1.85, 1.95, 1.3]))        # upper arm
        chains.append(([el, el + 0.3 * fa, wr], [1.35, 1.5, 0.95]))          # forearm
        dirn = fa / np.linalg.norm(fa)
        side = np.cross(dirn, Z)
        side /= np.linalg.norm(side) + 1e-9
        HR = np.stack([dirn, side, np.cross(dirn, side)], 1)
        E(wr + dirn * 1.9, (2.1, 1.25, 0.65), HR)                            # open hand
        E(wr + dirn * 0.9 + side * 0.9, (1.0, 0.45, 0.45), HR)               # thumb
    # neck and head, the face turned down the line toward the nose
    head = P(6.6, -2.2, 37.4)
    chains.append(([neck0, P(5.6, -1.8, 35.0)], [1.75, 1.55]))
    E(head, (2.85, 2.45, 3.2))
    E(head + U * 1.6 - Z * 1.9 - V * 0.2, (1.55, 1.75, 1.3))                 # jaw and chin
    E(head + U * 2.75 - Z * 0.2 - V * 0.25, (0.85, 0.5, 0.8))                # nose
    E(head + U * 2.25 + Z * 1.05 - V * 0.2, (0.55, 1.75, 0.5))               # brow
    for s in (-1, 1):
        E(head - U * 0.2 + V * s * 2.35 - Z * 0.1, (0.7, 0.45, 1.0))         # ears
    E(head - U * 0.5 + Z * 0.9 + V * 0.1, (3.0, 2.65, 2.55))                 # hair
    for du, dz, dv in ((-2.4, 1.4, -1.2), (-2.8, 0.6, 0.0), (-2.4, 1.6, 1.2), (-3.0, -0.4, 0.8)):
        a = head + U * 1.4 + Z * 2.2 + V * dv * 0.6
        chains.append(([a, head + U * du + Z * dz + V * dv], [0.75, 0.55]))  # swept-back locks
    # a plumeria lei round the neck, lying on the collarbones, lower on the chest
    lei = []
    nc = neck0 + Z * 0.4
    for k in range(11):
        t = 2 * np.pi * k / 11
        hang = 0.5 - 0.5 * np.sin(t)                             # 1 at the front (toward the viewer)
        p = nc + U * 3.8 * np.cos(t) + V * 3.3 * np.sin(t) - Z * 1.6 * hang - V * 0.8 * hang
        nrm = 0.55 * (U * np.cos(t) + V * np.sin(t)) + 0.85 * Z - 0.4 * V * hang
        lei.append((p, nrm / np.linalg.norm(nrm), t))
    k = SURFER_SCALE

    def sc(p):
        return O + k * (np.asarray(p) - O)
    chains = [([sc(p) for p in pts], [r * k * BULK for r in rr]) for pts, rr in chains]
    ells = [(sc(c), R, tuple(r_ * k * BULK for r_ in r)) for c, R, r in ells]
    shorts = [([sc(p) for p in pts], [r * k * BULK for r in rr]) for pts, rr in shorts]
    lei = [(sc(p), n, t) for p, n, t in lei]
    return dict(chains=chains, ells=ells, shorts=shorts, lei=lei, board=(C, eu, ev, en))


def frame_ell(x, y, z, c, R, r):
    px, py, pz = x - c[0], y - c[1], z - c[2]
    a = px * R[0, 0] + py * R[1, 0] + pz * R[2, 0]
    b = px * R[0, 1] + py * R[1, 1] + pz * R[2, 1]
    g = px * R[0, 2] + py * R[1, 2] + pz * R[2, 2]
    return ell(a, b, g, *r)


def lei_sdf(x, y, z, lei):
    f = plumeria(1.3 * SURFER_SCALE, lift=0.5)
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
        d = np.minimum(d, f(u * cs - v * sn, u * sn + v * cs, w + 0.25))
    return d


def surfer_sdf():
    S = surfer_parts()
    C, eu, ev, en = S["board"]

    def f(x, y, z):
        u, v, w = local(x, y, z, C, eu, ev, en)
        d = np.minimum(board_sdf(u, v, w), peg_sdf(u, v, w))
        b = np.full(x.shape, 1e9)
        for pts, rr in S["chains"]:
            b = smin(b, capsule_chain(x, y, z, np.array(pts), np.array(rr)), 0.7)
        for c, R, r in S["ells"]:
            b = smin(b, frame_ell(x, y, z, c, R, r), 0.6)
        sh = np.full(x.shape, 1e9)
        for pts, rr in S["shorts"]:
            sh = smin(sh, capsule_chain(x, y, z, np.array(pts), np.array(rr)), 0.5)
        b = np.minimum(b, sh)                                    # a crisp hem where the shorts end
        b = np.minimum(b, lei_sdf(x, y, z, S["lei"]))
        return smin(d, b, 0.5)
    pts = np.array([p for pts_, _ in S["chains"] for p in pts_] + [C])
    L = BOARD["L"] / 2 + 4
    lo = np.minimum(pts.min(0) - 6, C - L)
    hi = np.maximum(pts.max(0) + 6, C + L)
    return f, (lo, hi)


# ------------------------------------------------------------------ build

def bounds(part="wave"):
    if part == "surfer":
        return surfer_sdf()[1]
    return np.array([-BASE[0] - 3, -BASE[1] - 3, -0.6]), np.array([BASE[0] + 3, BASE[1] + 3, HMAX + 6])


def build_surfer(voxel, quiet=False, region=None):
    lo, hi = bounds("surfer") if region is None else (np.asarray(region[0]), np.asarray(region[1]))
    xs = np.arange(lo[0], hi[0], voxel, dtype=np.float32)
    ys = np.arange(lo[1], hi[1], voxel, dtype=np.float32)
    zs = np.arange(lo[2], hi[2], voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    f, _ = surfer_sdf()
    add(vol, xs, ys, zs, lo, hi, f, k=0, chunk=6)
    fill_voids(vol)
    drop_specks(vol, voxel, (lambda *a: None) if quiet else print)
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
