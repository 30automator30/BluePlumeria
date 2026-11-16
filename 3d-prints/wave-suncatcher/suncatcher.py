#!/usr/bin/env python3
"""Coast suncatcher, stained-glass style (clear resin, hand-painted), 150 mm.

A scene for the stained-glass engine (../stained-glass/stained_glass.py):
a jungle-topped cliff on the left in long vertical strips, white water
crashing at its foot, the sun setting on the horizon with rays fanning
across the whole sky, a paler headland behind the cliff, the sea in long
streaks with the sun's path on it, and hibiscus, plumeria, leaves and rocks
along the bottom.

Usage
  python3 suncatcher.py --guide-only               # paint guide only
  python3 suncatcher.py --voxel 0.2 --preview      # quick look
  python3 suncatcher.py --voxel 0.1                # final (decimated)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stained-glass"))
import stained_glass as sg  # noqa: E402

FRAME = sg.Frame(d=150.0)
OUT = "wave_suncatcher.stl"
HORIZON = 12.0
SUN = np.array([24.0, HORIZON]), 11.0

# suggested paint colours for the guide, by category
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


def draw(M):
    X, Y = M.X, M.Y
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
    sg.shatter(M, far, lambda x, y: "leaf_far" if y > H0 + 15 else "rock_far", 6.5, ang=np.radians(70), stretch=1.8,
            amp=1.0, seed=8)

    # ---- the cliff on the left, jungle on top
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
    sg.shatter(M, green, "leaf_dk", 5.0, ang=np.radians(35), stretch=2.0, amp=1.2, seed=3)
    for x0, y0, L, ang in jungle_x:
        sg.leaf(M, (x0, y0), ang, L, L * 0.42, 0.1, vein=4.5, lean=0.8, clip=Y < top + 6)

    # ---- the sea: long streaks, larger toward the viewer
    sea = (~sky) & ~cliff & (M.lab == 0)

    def persp(u, v):
        dep = np.clip(H0 - v, 0, None)
        k = 1.0 / (1.0 + dep / 60.0)
        return u * k, -np.log1p(dep / 60.0) * 60.0

    def sea_cat(x, y):
        dep = H0 - y
        return "sea_far" if dep < 9 else ("sea_mid" if dep < 26 else "sea_near")
    sg.shatter(M, sea, sea_cat, 4.4, stretch=3.0, amp=1.2, seed=4, tf=persp)
    # sun path on the water
    glint = sea & (np.abs(X - S[0] - 0.6 * np.sin(Y / 1.5)) < 1.8 + 0.22 * (H0 - Y)) & (Y > H0 - 14)
    sg.shatter(M, glint, "glint", 2.4, stretch=2.5, amp=0.5, seed=5, tf=persp)

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
    sg.shatter(M, foam, lambda x, y: "foam" if rng.random() < 0.7 else "foam_sh", 6.0, ang=np.radians(-60),
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
        sg.leaf(M, base, ang, L, Wd, bend, vein=5.0, lean=0.9)
    # rocks at the cliff foot, under the surf line
    for k, (c, R) in enumerate([((-50, -46), 9), ((-36, -52), 7), ((-58, -58), 6)]):
        P = sg.blob_poly(np.array(c, float), R, jag=0.12, seed=k)
        sg.shatter(M, sg.poly_sdf(X, Y, P) < 0, "rock", 5.0, amp=0.6, seed=20 + k)

    sg.plumeria(M, (44, -20), 9.5, 0.6)
    sg.hibiscus(M, (52, -40), 15.0, 0.9, "hib_or", "hib_or_c")
    sg.hibiscus(M, (22, -50), 18.5, 0.3, "hibiscus", "hib_c")
    sg.plumeria(M, (50, -60), 10.0, 1.2, "pink")
    sg.plumeria(M, (2, -62), 10.5, 0.2)
    sg.plumeria(M, (-28, -58), 9.0, 0.9, "pink")
    sg.plumeria(M, (-48, -32), 8.5, 0.4)


if __name__ == "__main__":
    sg.main(sys.modules[__name__])
