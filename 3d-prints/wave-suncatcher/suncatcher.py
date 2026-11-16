#!/usr/bin/env python3
"""Stained-glass wave suncatcher (clear resin, Saturn 4 Ultra), 150 mm.

Printed clear, then painted by hand with glass paint. The picture is a
flat clear pane crossed by raised "lead" lines; the lines close every cell,
so each cell holds its own colour like a little dam.

The scene, in a hand-cut glass style: a jungle-topped cliff on the left,
white water crashing at its foot, the sun setting on the horizon with rays,
the sea in long streaks, and hibiscus, plumeria and leaves at the bottom.
Large areas are cut into irregular pieces (a warped, stretched Voronoi
pattern) that lie along the way each thing flows.

How it is made: the scene is drawn as a map of labelled regions (painter's
order, later shapes on top). Connected regions become cells; cells too small
or too thin to paint are merged into a neighbour. Lead lines run along every
boundary between cells, plus a few free lines (the hibiscus stamen). The
solid is the pane (back flat) plus the lines (rounded tops, straight walls),
a rounded frame ring, and a hanging loop at the top.

Usage
  python3 suncatcher.py --voxel 0.2 --preview      # quick look
  python3 suncatcher.py --voxel 0.08               # final (decimated)
  python3 suncatcher.py --guide-only               # paint guide images only
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("fractal-plumeria", "plumeria-ring", "nalu"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
from fractal_plumeria import mesh_volume  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402
import nalu  # noqa: E402

D = 150.0                      # outer diameter
RING_W, RING_T = 5.0, 4.0      # frame width and thickness
R_OUT = D / 2
R_IN = R_OUT - RING_W
PANE_T = 1.6                   # the clear pane
LINE_W, LINE_H = 1.0, 1.0      # lead lines: width, height above the pane
LOOP = dict(r=4.0, tube=2.0)   # hanging loop: hole radius, tube radius

HORIZON = 12.0
SUN = np.array([24.0, HORIZON]), 11.0
WAVE = dict(H=60.0, foot=-30.0, x=-9.0, sx=1.75)

MIN_AREA = 5.0                 # mm^2: smaller cells are merged
MIN_HALF = 1.15                # mm: cells thinner than 2x this are merged

# paint colours for the guide (glass-paint names), by category
PALETTE = {
    "sky_hi": [(0.35, 0.55, 0.85), (0.45, 0.62, 0.88), (0.30, 0.45, 0.78)],
    "sky_mid": [(0.62, 0.70, 0.92), (0.95, 0.70, 0.55), (0.80, 0.62, 0.80)],
    "sky_lo": [(0.99, 0.80, 0.45), (0.98, 0.68, 0.42), (0.97, 0.86, 0.60)],
    "sky_sun": [(1.0, 0.88, 0.45), (0.99, 0.78, 0.38)],
    "ray": [(1.0, 0.86, 0.35), (1.0, 0.74, 0.30), (0.99, 0.93, 0.55)],
    "sun": [(1.0, 0.95, 0.55)],
    "rock": [(0.60, 0.34, 0.18), (0.72, 0.44, 0.22), (0.50, 0.28, 0.16), (0.80, 0.55, 0.30)],
    "rock_lit": [(0.86, 0.58, 0.30), (0.92, 0.66, 0.36), (0.80, 0.50, 0.26)],
    "rock_far": [(0.62, 0.48, 0.55), (0.70, 0.55, 0.60)],
    "leaf_far": [(0.42, 0.58, 0.45), (0.50, 0.65, 0.48)],
    "glow": [(1.0, 0.84, 0.30)],
    "rock_dk": [(0.48, 0.26, 0.14), (0.62, 0.36, 0.20), (0.70, 0.42, 0.22)],
    "leaf": [(0.16, 0.55, 0.22), (0.30, 0.68, 0.24), (0.22, 0.48, 0.18)],
    "leaf_dk": [(0.10, 0.40, 0.18), (0.18, 0.52, 0.20), (0.36, 0.62, 0.20)],
    "sea_far": [(0.18, 0.30, 0.66), (0.24, 0.36, 0.72), (0.30, 0.30, 0.62)],
    "sea_mid": [(0.12, 0.42, 0.72), (0.16, 0.55, 0.78), (0.22, 0.40, 0.70)],
    "sea_near": [(0.10, 0.62, 0.72), (0.18, 0.70, 0.78), (0.12, 0.50, 0.70), (0.30, 0.76, 0.80)],
    "glint": [(1.0, 0.88, 0.50), (0.98, 0.95, 0.75)],
    "foam": [(0.97, 0.98, 0.99), (0.92, 0.95, 0.98)],
    "foam_sh": [(0.78, 0.86, 0.94)],
    "plumeria": [(0.99, 0.98, 0.92)],
    "pink": [(0.96, 0.55, 0.72), (0.94, 0.48, 0.66)],
    "plum_c": [(1.0, 0.82, 0.30)],
    "hibiscus": [(0.88, 0.14, 0.20), (0.95, 0.26, 0.30)],
    "hib_c": [(0.55, 0.05, 0.12)],
    "hib_or": [(0.99, 0.62, 0.18), (1.0, 0.72, 0.25)],
    "hib_or_c": [(0.85, 0.20, 0.15)],
}


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def smax(a, b, k):
    return -smin(-a, -b, k)


# ---------------------------------------------------------------- shapes --

def wave_poly():
    W = WAVE
    p = nalu.BARREL.astype(float).copy()          # the barrel profile, a clean lip (no tears)
    p[:, 0] *= 0.5 + 0.5 * W["H"] / nalu.HMAX
    p[:, 1] *= W["H"]
    P = nalu.chaikin(np.vstack([p, p[:1]]))
    return np.stack([W["x"] + P[:, 0] * W["sx"], W["foot"] + P[:, 1]], 1)


def petal_poly(R, a, n=40, ruffle=0.0, width=0.47):
    """One obovate petal pointing along angle a, offset pinwheel-fashion."""
    t = np.linspace(0, 1, n)
    wid = width * R * np.minimum(t / 0.62, 1) ** 0.8 * np.clip(1 - (np.maximum(t - 0.62, 0) / 0.38) ** 2.2, 0, 1)
    u = t * R
    left = np.stack([u, wid * 1.1], 1)
    right = np.stack([u, -wid * 0.9], 1)[::-1]
    P = np.vstack([left, right])
    if ruffle:
        # a frilled outer edge: bump the far end in and out
        far = np.clip((P[:, 0] / R - 0.7) / 0.3, 0, 1)
        ang = np.arctan2(P[:, 1], P[:, 0] + 1e-9)
        P[:, 0] += ruffle * R * far * np.sin(ang * 9)
    P = np.vstack([P, P[:1]])
    off = 0.15 * R
    c = np.array([-off * np.sin(a), off * np.cos(a)])
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return P @ Rm.T + c


def leaf_poly(base, ang, L, W, bend=0.15, n=40):
    """A ti leaf: long, pointed both ends, gently bent. Returns the two
    halves (split along the midrib) as polygons."""
    t = np.linspace(0, 1, n)
    w = W / 2 * np.sin(np.pi * t ** 0.8) ** 0.9
    mid = np.stack([t * L, bend * L * np.sin(np.pi * t) * t], 1)
    nrm = np.array([0.0, 1.0])
    left = mid + nrm * w[:, None]
    right = mid - nrm * w[:, None]
    Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    halves = []
    for side in (left, right):
        P = np.vstack([mid, side[::-1], mid[:1]])
        halves.append(P @ Rm.T + base)
    return halves


# --------------------------------------------------------------- the map --

class Map:
    """Label map on a grid: each pixel gets a region id; ids carry a
    category name for the paint guide."""

    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.lab = np.zeros(X.shape, np.int32)
        self.cat = ["none"]

    def paint(self, mask, cat):
        self.lab[mask] = len(self.cat)
        self.cat.append(cat)

    def paint_split(self, mask, keys, cat):
        """Paint one region split into sub-cells by an integer key array."""
        u = np.unique(keys[mask])
        for k in u:
            self.paint(mask & (keys == k), cat)


def local(X, Y, C, a):
    """Coordinates along (u) and across (v) a direction a from point C."""
    dx, dy = X - C[0], Y - C[1]
    return dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)


def warp(X, Y, amp, s=1.0):
    """A gentle domain warp, so cut lines bow like hand-cut glass."""
    wx = amp * (np.sin(Y / (6.3 * s) + 1.3) + 0.6 * np.sin((X + Y) / (4.1 * s)))
    wy = amp * (np.sin(X / (5.7 * s) + 0.4) + 0.6 * np.sin((X - Y) / (3.7 * s) + 2.0))
    return X + wx, Y + wy


def shatter(M, mask, cat, spacing, ang=0.0, stretch=1.0, amp=1.0, seed=0, tf=None):
    """Cut a region into irregular glass pieces: a Voronoi pattern of
    well-spaced seeds, measured in a warped, stretched space (long pieces
    lie along angle ang). tf, if given, maps (u, v) to a space where the
    pieces should be even (e.g. perspective for the sea). cat may be a
    function of the piece's centre giving its category."""
    from scipy.spatial import cKDTree
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return
    X, Y = M.X.ravel()[idx], M.Y.ravel()[idx]
    Xw, Yw = warp(X, Y, amp)
    u = Xw * np.cos(ang) + Yw * np.sin(ang)
    v = -Xw * np.sin(ang) + Yw * np.cos(ang)
    if tf is not None:
        u, v = tf(u, v)
    Q = np.stack([u / stretch, v], 1)
    rng = np.random.default_rng(seed)
    cand = rng.permutation(len(idx))[:40000]
    seeds = []
    grid = {}
    cs = spacing / np.sqrt(2)
    for c in cand:
        q = Q[c]
        gx, gy = int(q[0] // cs), int(q[1] // cs)
        ok = True
        for i in range(gx - 2, gx + 3):
            for j in range(gy - 2, gy + 3):
                s = grid.get((i, j))
                if s is not None and (s[0] - q[0]) ** 2 + (s[1] - q[1]) ** 2 < spacing ** 2:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[(gx, gy)] = q
            seeds.append(c)
    S = Q[seeds]
    _, near = cKDTree(S).query(Q)
    for k in np.unique(near):
        m = np.zeros(M.X.size, bool)
        sel = near == k
        m[idx[sel]] = True
        c = cat(X[sel].mean(), Y[sel].mean()) if callable(cat) else cat
        M.paint(m.reshape(M.X.shape), c)


def blob_poly(C, R, n=60, jag=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ph = rng.uniform(0, 6.3, 3)
    rad = R * (1 + jag * (np.sin(3 * t + ph[0]) * 0.5 + np.sin(5 * t + ph[1]) * 0.3 + np.sin(8 * t + ph[2]) * 0.2))
    P = np.stack([C[0] + rad * np.cos(t), C[1] + rad * np.sin(t)], 1)
    return np.vstack([P, P[:1]])


def hib_petal(C, R, a, w=np.radians(42), ruffle=0.05, n=50):
    """A hibiscus petal: a broad fan from the centre, rounded and frilled."""
    ph = np.linspace(-w, w, n)
    s = ph / w
    rad = R * (0.62 + 0.38 * np.sqrt(np.clip(1 - s ** 2, 0, 1))) * (1 + ruffle * np.sin(ph * 26 + a * 3))
    rad *= 1 - 0.08 * np.exp(-((s) / 0.12) ** 2)                 # a small notch at the tip
    P = np.stack([C[0] + rad * np.cos(a + ph), C[1] + rad * np.sin(a + ph)], 1)
    P = np.vstack([[C], P, [C]])
    return P


def design(res):
    """The scene on a grid of pitch res. Returns xs, the region map
    (0 = outside the pane), each region's category, and the distance to the
    free lines."""
    xs = np.arange(-R_IN - 1, R_IN + 1 + res, res)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    M = Map(X, Y)
    r = np.hypot(X, Y)
    dfree = np.full(X.shape, 1e9)
    H0 = HORIZON
    S, sr = SUN

    # ---- sky: rays fanning out from the sun across the whole sky, each ray
    # cut into a few pieces by wavy arcs; warm near the sun, cooler out
    sky = Y >= H0
    rs = np.hypot(X - S[0], Y - S[1])
    th = np.arctan2(Y - S[1], X - S[0])
    rng = np.random.default_rng(5)
    widths = rng.uniform(9, 16, 30)
    edges = np.radians(np.cumsum(np.r_[-1.0, widths]) - 0.0)
    tw = th + 0.035 * np.sin(rs / 5.0 + 1.0) + 0.02 * np.sin(rs / 2.3)
    ray = np.digitize(tw, edges)
    arc = np.zeros(X.shape, int)
    for k in range(len(edges) + 1):
        m = ray == k
        cuts = sr + np.cumsum(rng.uniform(8, 13, 6) * np.array([1.0, 1.15, 1.3, 1.45, 1.6, 1.8]))
        wob = 1.6 * np.sin(th[m] * 9 + k)
        arc[m] = np.digitize(rs[m] + wob, cuts)

    def sky_cat_k(a_):
        return ["ray", "ray", "sky_lo", "sky_lo", "sky_mid", "sky_hi", "sky_hi"][min(a_, 6)]
    for a_ in np.unique(arc[sky]):
        M.paint_split(sky & (arc == a_), ray, sky_cat_k(a_))
    M.paint_split(sky & (rs < sr + 4 + 1.0 * np.sin(th * 11)), np.floor(np.degrees(th) / 30).astype(int), "glow")
    M.paint_split(sky & (rs < sr), np.digitize(Y - H0, [sr * 0.45]), "sun")

    # ---- a far headland behind the cliff (paler, for depth)
    far_top = H0 + 24 - 0.5 * np.clip(X + 30, 0, None) + 2.5 * np.abs(np.sin(X / 3.1)) - 0.012 * (X + 30) ** 2 * (X < -30)
    far = sky & (Y < far_top) & (X < 20)
    shatter(M, far, lambda x, y: "leaf_far" if y > H0 + 15 else "rock_far", 6.5, ang=np.radians(70), stretch=1.8,
            amp=1.0, seed=8)

    # ---- the cliff on the left: jagged crags, lit faces and shadowed
    # buttresses, jungle on top
    def tri(t):
        return 2 * np.abs(t - np.floor(t + 0.5))
    # long, gently curving vertical strips running down the cliff, like
    # streaked brown glass, each cut once or twice on a slant
    top = 42 - 0.12 * (X + 70) + 2.5 * np.sin(X / 8.0 + 0.5) + 1.2 * np.sin(X / 3.7)
    face = -25 + 0.30 * (40 - Y) + 1.6 * np.sin(Y / 6.0)
    cliff = (X < face) & (Y < top) & (Y > -34 + 6 * np.sin((X + 70) / 18.0) + 0.25 * (X + 70))
    sx = X + 0.22 * (40 - Y) + 2.2 * np.sin(Y / 11.0) + 1.0 * np.sin(Y / 5.3 + X / 9.0)
    srng = np.random.default_rng(21)
    edges_x = np.cumsum(np.r_[-90.0, srng.uniform(7.0, 11.0, 14)])
    strip = np.digitize(sx, edges_x)
    tones = ["rock_lit", "rock", "rock_dk"]
    for k in np.unique(strip[cliff]):
        m = cliff & (strip == k)
        ys = Y[m]
        y0, y1 = ys.min(), ys.max()
        ncut = 1 if y1 - y0 < 35 else 2
        cuts = np.sort(srng.uniform(0.25, 0.75, ncut)) * (y1 - y0) + y0
        slope = srng.choice([-1, 1]) * srng.uniform(0.5, 1.0)
        seg = np.digitize(Y + slope * (X - X[m].mean()), cuts)
        for j in range(ncut + 1):
            M.paint(m & (seg == j), tones[srng.integers(3)])
    green = cliff & ((Y > top - 11 - 3 * np.sin(X / 4.0)) | ((X > face - 5 - 2 * np.sin(Y / 3)) & (Y > 24) & (Y < top)))
    jungle_x = [(-64, 47, 11, 60), (-54, 50, 12, 85), (-44, 49, 11, 70), (-34, 45, 10, 30), (-60, 38, 9, 150),
                (-27, 37, 8, -20), (-49, 41, 9, 120), (-22, 30, 7, -40)]
    shatter(M, green, "leaf_dk", 5.0, ang=np.radians(35), stretch=2.0, amp=1.2, seed=3)
    for k, (x0, y0, L, ang) in enumerate(jungle_x):
        a = np.radians(ang)
        u, v = local(X, Y, np.array([x0, y0], float), a)
        for h in leaf_poly(np.array([x0, y0], float), a, L, L * 0.42, 0.1):
            m = nalu.poly_sdf(X, Y, h) < 0
            M.paint_split(m & (Y < top + 6), np.floor((u - 0.8 * np.abs(v)) / 4.5).astype(int), "leaf")

    # ---- the sea: long streaks, larger toward the viewer
    sea = (~sky) & ~cliff & (M.lab == 0)

    def persp(u, v):
        dep = np.clip(H0 - v, 0, None)
        k = 1.0 / (1.0 + dep / 60.0)
        return u * k, -np.log1p(dep / 60.0) * 60.0

    def sea_cat(x, y):
        dep = H0 - y
        return "sea_far" if dep < 9 else ("sea_mid" if dep < 26 else "sea_near")
    shatter(M, sea, sea_cat, 4.4, stretch=3.0, amp=1.2, seed=4, tf=persp)
    # sun path on the water
    glint = sea & (np.abs(X - S[0] - 0.6 * np.sin(Y / 1.5)) < 1.8 + 0.22 * (H0 - Y)) & (Y > H0 - 14)
    shatter(M, glint, "glint", 2.4, stretch=2.5, amp=0.5, seed=5, tf=persp)

    # ---- surf: a mass of white water crashing on the cliff foot and
    # rolling in toward the viewer (billowed edge, cut into big pieces), and
    # a thin breaking crest out on the sea
    rng = np.random.default_rng(11)
    pth = np.array([[-26, 4], [-16, -6], [-9, -16], [-8, -27], [-12, -39], [-18, -50], [-22, -64]], float)
    seg = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pth, axis=0), axis=1)])
    foam = np.zeros(X.shape, bool)
    for s_ in np.arange(0, seg[-1], 2.2):
        p = np.array([np.interp(s_, seg, pth[:, 0]), np.interp(s_, seg, pth[:, 1])])
        wid = 15.0 - 5.0 * s_ / seg[-1]
        for _ in range(2):
            R = rng.uniform(0.35, 0.6) * wid
            c = p + rng.normal(0, wid * 0.35, 2)
            foam |= np.hypot(X - c[0], Y - c[1]) < R
    foam &= (Y < H0 + 1.5) & ((X > face - 1.5) | (Y < -32))
    shatter(M, foam, lambda x, y: "foam" if rng.random() < 0.7 else "foam_sh", 6.0, ang=np.radians(-60),
            stretch=1.6, amp=1.6, seed=6)
    # the crest: a white streak along the swell with a scalloped top
    cx = np.array([-18, 0, 18, 36, 54, 72.0])
    cy = np.array([7, 4.5, 3, 0.5, 2, -1.0])
    yc = np.interp(X, cx, cy)
    th_ = 1.2 + 1.6 * np.clip((X + 18) / 30, 0, 1) * (1 - np.clip((X - 40) / 32, 0, 1)) + 0.4
    scal = 0.9 * np.abs(np.sin(X / 2.6))
    crest = (Y < yc + th_ / 2 + scal) & (Y > yc - th_ / 2) & (X > -20) & (M.lab > 0)
    M.paint_split(crest, np.floor((X + 1.5 * np.sin(Y)) / 9.0).astype(int), "foam")

    # ---- flowers and leaves, bottom right, and a few at the cliff foot
    leaves = [((62, -40), 150, 34, 13, -0.15), ((50, -66), 115, 34, 13, 0.12), ((22, -72), 75, 32, 12, -0.10),
              ((68, -20), 185, 26, 11, 0.10), ((36, -36), 40, 28, 11, 0.15),
              ((-60, -40), 40, 28, 12, 0.12), ((-40, -66), 70, 30, 12, -0.12), ((-66, -24), 10, 24, 10, 0.1)]
    for base, ang, L, Wd, bend in leaves:
        a = np.radians(ang)
        u, v = local(X, Y, np.array(base, float), a)
        for h in leaf_poly(np.array(base, float), a, L, Wd, bend):
            m = nalu.poly_sdf(X, Y, h) < 0
            M.paint_split(m, np.floor((u - 0.9 * np.abs(v)) / 5.0).astype(int), "leaf")
    # rocks at the cliff foot, under the surf line
    for k, (c, R) in enumerate([((-50, -46), 9), ((-36, -52), 7), ((-58, -58), 6)]):
        P = blob_poly(np.array(c, float), R, jag=0.12, seed=k)
        shatter(M, nalu.poly_sdf(X, Y, P) < 0, "rock", 5.0, amp=0.6, seed=20 + k)

    def hibiscus(C, R, rot, cat, eye):
        nonlocal dfree
        C = np.array(C, float)
        for k in range(5):
            a = rot + 2 * np.pi * k / 5
            P = hib_petal(C, R, a)
            m = nalu.poly_sdf(X, Y, P) < 0
            rr = np.hypot(X - C[0], Y - C[1])
            u, v = local(X, Y, C, a)
            inner = rr < R * (0.36 + 0.06 * np.sin(np.arctan2(v, u) * 9))
            side = (v > 0).astype(int)
            M.paint(m & ~inner, cat)
            M.paint(m & inner, eye)
            for dv in (0.0,):                         # the petal's vein
                aa = a + dv * np.radians(30)
                dfree = np.minimum(dfree, seg_dist(X, Y, C + 0.42 * R * np.array([np.cos(aa), np.sin(aa)]),
                                                   C + 0.74 * R * np.array([np.cos(aa), np.sin(aa)])))
        M.paint(np.hypot(X - C[0], Y - C[1]) < 0.12 * R, eye)
        # the staminal column, curving out, with its pollen
        sa = rot + np.radians(20)
        t = np.linspace(0.12, 1.05, 12)
        pts = C + R * np.stack([t * np.cos(sa + 0.25 * t), t * np.sin(sa + 0.25 * t)], 1)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            dfree = np.minimum(dfree, seg_dist(X, Y, p0, p1))
        tip = pts[-1]
        for k in range(6):
            b = sa + np.radians(-75 + 30 * k)
            q = tip + 0.12 * R * np.array([np.cos(b), np.sin(b)])
            dfree = np.minimum(dfree, np.hypot(X - q[0], Y - q[1]) - 0.55)

    def plumeria(C, R, rot, cat="plumeria"):
        C = np.array(C, float)
        for k in range(5):
            a = 2 * np.pi * k / 5 + rot
            P = petal_poly(R, a, width=0.47) + C
            m = nalu.poly_sdf(X, Y, P) < 0
            off = 0.15 * R
            c = C + np.array([-off * np.sin(a), off * np.cos(a)])
            u, v = local(X, Y, c, a)
            throat = np.hypot(X - C[0], Y - C[1]) < 0.40 * R * (1 + 0.12 * np.sin(4 * np.arctan2(v, u)))
            M.paint(m & ~throat, cat)
            M.paint(m & throat, "plum_c")
        M.paint(np.hypot(X - C[0], Y - C[1]) < 0.13 * R, "plum_c")

    plumeria((44, -20), 9.5, 0.6)
    hibiscus((52, -40), 15.0, 0.9, "hib_or", "hib_or_c")
    hibiscus((22, -50), 18.5, 0.3, "hibiscus", "hib_c")
    plumeria((50, -60), 10.0, 1.2, "pink")
    plumeria((2, -62), 10.5, 0.2)
    plumeria((-28, -58), 9.0, 0.9, "pink")
    plumeria((-48, -32), 8.5, 0.4)

    M.lab[r > R_IN + 0.4] = 0
    dfree[r > R_IN] = 1e9
    return xs, M.lab, M.cat, dfree


def seg_dist(X, Y, a, b):
    e = b - a
    h = np.clip(((X - a[0]) * e[0] + (Y - a[1]) * e[1]) / (e @ e), 0, 1)
    return np.hypot(X - a[0] - h * e[0], Y - a[1] - h * e[1])


# ------------------------------------------------------------- the cells --

def cells(lab, res):
    """Split regions into connected cells, then merge cells too small or too
    thin to paint into the neighbour they share the most boundary with."""
    cid = np.zeros(lab.shape, np.int32)
    n = 0
    for v in np.unique(lab):
        if v == 0:
            continue
        cc, k = ndi.label(lab == v)
        cid[cc > 0] = cc[cc > 0] + n
        n += k
    src = np.zeros(n + 1, np.int32)               # the region each cell came from
    src[cid.ravel()] = lab.ravel()
    for _ in range(12):
        ids, area = np.unique(cid, return_counts=True)
        area = area * res * res
        # thickness: largest inscribed half-width per cell
        half = np.zeros(cid.max() + 1)
        dt = ndi.distance_transform_edt(np.pad(boundary(cid) == 0, 1, constant_values=False))[1:-1, 1:-1] * res
        np.maximum.at(half, cid.ravel(), dt.ravel())
        bad = [i for i, a in zip(ids, area) if i > 0 and (a < MIN_AREA or half[i] < MIN_HALF)]
        if not bad:
            break
        # shared boundary lengths between neighbours
        pairs = []
        for A, B in ((cid[1:, :], cid[:-1, :]), (cid[:, 1:], cid[:, :-1])):
            m = (A != B) & (A > 0) & (B > 0)
            pairs.append(np.stack([A[m], B[m]], 1))
        pairs = np.vstack(pairs)
        pairs = np.vstack([pairs, pairs[:, ::-1]])
        key, cnt = np.unique(pairs, axis=0, return_counts=True)
        remap = np.arange(cid.max() + 1)
        bad = sorted(bad, key=lambda i: area[np.searchsorted(ids, i)])
        taken = set()
        for i in bad:
            if i in taken:
                continue
            m = key[:, 0] == i
            if not m.any():
                remap[i] = 0
                continue
            j = key[m][np.argmax(cnt[m]), 1]
            remap[i] = j
            taken.add(j)
        cid = remap[cid]
    # renumber
    u, inv = np.unique(cid, return_inverse=True)
    newsrc = np.array([src[x] if x > 0 else 0 for x in u])
    cid = inv.reshape(cid.shape)
    if u[0] != 0:
        cid += 1
        newsrc = np.concatenate([[0], newsrc])
    return cid, newsrc


def boundary(cid):
    """Pixels on the edge of a cell (a 4-neighbour belongs to another)."""
    b = np.zeros(cid.shape, bool)
    b[1:, :] |= cid[1:, :] != cid[:-1, :]
    b[:-1, :] |= cid[1:, :] != cid[:-1, :]
    b[:, 1:] |= cid[:, 1:] != cid[:, :-1]
    b[:, :-1] |= cid[:, 1:] != cid[:, :-1]
    return b


def line_distance(cid, dfree, res):
    """Distance to the nearest lead line centre: cell boundaries (both
    sides of a boundary are marked, so the centre is half a pixel in) and
    the free lines."""
    b = boundary(cid) & (cid > 0)
    d = ndi.distance_transform_edt(~b) * res
    return np.minimum(d, dfree)


# ---------------------------------------------------------------- build --

def build(voxel):
    t0 = time.time()
    xs, lab, cat, dfree = design(voxel)
    cid, src = cells(lab, voxel)
    ncell = cid.max()
    print(f"  {ncell} cells, design {time.time() - t0:.0f}s")
    dline = line_distance(cid, dfree, voxel)
    # extend the 2D fields to the full grid (frame and loop above it)
    lo = np.array([-R_OUT - 1.0, -R_OUT - 1.0, -0.4])
    ys_top = R_OUT + 2 * LOOP["r"] + 2 * LOOP["tube"] + 2
    gx = np.arange(lo[0], R_OUT + 1.0, voxel)
    gy = np.arange(lo[1], ys_top, voxel)
    gz = np.arange(lo[2], RING_T + 0.4, voxel)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    r = np.hypot(X, Y)
    # resample dline onto the full grid (same pitch, offset origin)
    ix = np.clip(np.round((gx - xs[0]) / voxel).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((gy - xs[0]) / voxel).astype(int), 0, len(xs) - 1)
    DL = dline[np.ix_(ix, iy)]
    hw, rr = LINE_W / 2, 0.45                       # half-width, top rounding
    top = PANE_T + LINE_H
    open_ = R_IN + 0.5 - r                          # >0 inside the frame's opening
    ring_r = np.abs(r - (R_OUT - RING_W / 2)) - RING_W / 2
    lc = np.array([0.0, R_OUT + LOOP["r"] + LOOP["tube"] * 0.3, RING_T / 2])
    q = np.hypot(X - lc[0], Y - lc[1]) - LOOP["r"] - LOOP["tube"]
    vol = np.empty((len(gx), len(gy), len(gz)), np.float32)
    for k, z in enumerate(gz):
        pane = np.maximum(np.maximum(z - PANE_T, -z), -open_)
        # a line: a rounded box in (distance across, height)
        qx, qz = DL - (hw - rr), z - (top - rr)
        line = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - rr
        line = np.maximum(np.maximum(line, -z), -open_)
        frame = smax(ring_r, np.abs(z - RING_T / 2) - RING_T / 2, 1.2)
        d = np.minimum(np.minimum(pane, line), frame)
        d = smin(d, np.hypot(q, z - lc[2]) - LOOP["tube"], 0.8)
        vol[:, :, k] = d
    print(f"  built {time.time() - t0:.0f}s")
    return vol, (gx[0], gy[0], gz[0]), (xs, cid, src, cat, dline)


def guide(xs, cid, src, cat, dline, out, numbered=True):
    """Paint guide: the cells coloured as suggested, lines dark."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    n = cid.max() + 1
    cols = np.ones((n, 3))
    for i in range(1, n):
        c = cat[src[i]]
        pal = PALETTE.get(c, [(0.8, 0.8, 0.8)])
        base = np.array(pal[rng.integers(len(pal))])
        cols[i] = np.clip(base * (0.93 + 0.12 * rng.random()), 0, 1)
    img = cols[cid]
    img[cid == 0] = (1, 1, 1)
    hw = LINE_W / 2
    img[dline < hw] = (0.18, 0.18, 0.2)
    r = np.hypot(*np.meshgrid(xs, xs, indexing="ij"))
    img[(r > R_IN) & (r < R_OUT)] = (0.18, 0.18, 0.2)
    img = np.transpose(img, (1, 0, 2))[::-1]
    fig, ax = plt.subplots(1, 2 if numbered else 1, figsize=(16 if numbered else 8, 8), squeeze=False)
    ax = ax.ravel()
    ext = (xs[0], xs[-1], xs[0], xs[-1])
    ax[0].imshow(img, extent=ext)
    ax[0].set_title("painted (suggested colours)", fontsize=13)
    if numbered:
        line = np.ones(cid.shape + (3,))
        line[dline < hw] = 0.2
        line[(r > R_IN) & (r < R_OUT)] = 0.2
        ax[1].imshow(np.transpose(line, (1, 0, 2))[::-1], extent=ext)
        ax[1].set_title("lead lines (as printed)", fontsize=13)
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.2)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--guide-only", action="store_true")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    t0 = time.time()
    if args.guide_only:
        xs, lab, cat, dfree = design(args.voxel)
        cid, src = cells(lab, args.voxel)
        print(f"{cid.max()} cells")
        guide(xs, cid, src, cat, line_distance(cid, dfree, args.voxel), os.path.join(args.outdir, "paint_guide.png"))
        return
    vol, origin, g = build(args.voxel)
    mesh = mesh_volume(vol, args.voxel, origin)
    del vol
    if not args.preview:
        mesh = decimate(mesh, 0.004)
    print(f"suncatcher: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, {time.time() - t0:.0f}s")
    out = os.path.join(args.outdir, "wave_suncatcher.stl")
    mesh.export(out)
    guide(*g, os.path.join(args.outdir, "paint_guide.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
