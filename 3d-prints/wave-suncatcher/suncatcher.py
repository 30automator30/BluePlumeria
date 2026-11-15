#!/usr/bin/env python3
"""Stained-glass wave suncatcher (clear resin, Saturn 4 Ultra), 150 mm.

Printed clear, then painted by hand with glass paint. The picture is a
flat clear pane crossed by raised "lead" lines; the lines close every cell,
so each cell holds its own colour like a little dam.

The scene: a curling wave breaking to the left (the barrel profile from
nalu/), white water on its lip, a sun setting on the horizon at the right
with rays across the sky, the sea broken into shards, and plumeria,
hibiscus and ti leaves along the bottom.

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
LINE_W, LINE_H = 1.3, 1.1      # lead lines: width, height above the pane
LOOP = dict(r=4.0, tube=2.0)   # hanging loop: hole radius, tube radius

HORIZON = 9.0
SUN = np.array([33.0, HORIZON]), 12.0
WAVE = dict(H=60.0, foot=-30.0, x=-9.0, sx=1.75)

MIN_AREA = 9.0                 # mm^2: smaller cells are merged
MIN_HALF = 1.5                 # mm: cells thinner than 2x this are merged

# paint colours for the guide (glass-paint names), by category
PALETTE = {
    "sky": [(0.98, 0.80, 0.35), (0.99, 0.66, 0.30), (0.97, 0.52, 0.30), (0.90, 0.42, 0.45), (0.62, 0.40, 0.62)],
    "sun": [(1.0, 0.93, 0.45)],
    "glow": [(1.0, 0.82, 0.30)],
    "sea": [(0.10, 0.45, 0.70), (0.14, 0.58, 0.78), (0.20, 0.36, 0.66), (0.12, 0.66, 0.72)],
    "wave": [(0.06, 0.55, 0.62), (0.10, 0.66, 0.70), (0.16, 0.48, 0.70), (0.30, 0.74, 0.76)],
    "tube": [(0.62, 0.88, 0.86)],
    "foam": [(0.96, 0.97, 0.98), (0.88, 0.94, 0.97)],
    "leaf": [(0.16, 0.55, 0.22), (0.30, 0.68, 0.24)],
    "plumeria": [(0.99, 0.97, 0.90)],
    "plum_c": [(1.0, 0.80, 0.25)],
    "hibiscus": [(0.88, 0.16, 0.22), (0.95, 0.30, 0.36)],
    "hib_c": [(0.55, 0.05, 0.12)],
    "pink": [(0.96, 0.55, 0.72)],
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


def design(res):
    """The scene on a grid of pitch res. Returns xs, ys, the cell id map
    (0 = outside the pane), each cell's category, and the distance to the
    free lines."""
    xs = np.arange(-R_IN - 1, R_IN + 1 + res, res)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    M = Map(X, Y)
    r = np.hypot(X, Y)

    # sky: rays from the sun, split by two wavy arcs
    S, sr = SUN
    rs = np.hypot(X - S[0], Y - S[1])
    th = np.degrees(np.arctan2(Y - S[1], X - S[0]))
    ray = np.floor((th + 6) / 20.0).astype(int)
    ring = np.digitize(rs, [36.0, 62.0])
    sky = Y >= HORIZON
    M.paint_split(sky, ray * 10 + ring, "sky")
    # sun and its glow
    M.paint_split(sky & (rs < sr + 5), np.floor((th + 6) / 34.0).astype(int), "glow")
    M.paint(sky & (rs < sr), "sun")

    # sea: shards in bands that widen toward the viewer
    depth = HORIZON - Y
    wob = 1.2 * np.sin(X / 7.0 + 0.4 * depth) + 0.7 * np.sin(X / 3.1 - 1.0)
    edges = np.cumsum(np.linspace(5.5, 11.0, 10))
    band = np.digitize(depth + wob * np.sqrt(np.clip(depth, 0, None)) / 3.0, edges)
    shard = np.floor((X + 3.0 * band + 0.35 * depth + 9 * (band % 2)) / (16 + 1.5 * band)).astype(int)
    M.paint_split(~sky, band * 100 + shard, "sea")

    # the wave
    P = wave_poly()
    dw = nalu.poly_sdf(X, Y, P)
    inw = dw < 0
    # the middle of the barrel: of the open points under the lip, the one
    # farthest from the water
    above = np.flip(np.maximum.accumulate(np.flip(inw, 1), 1), 1)
    cand = (dw > 0) & above & (Y > WAVE["foot"] + 8) & (X < WAVE["x"] + 10)
    i = np.argmax(np.where(cand, dw, -1))
    C, room = np.array([X.ravel()[i], Y.ravel()[i]]), dw.ravel()[i]
    rr = np.hypot(X - C[0], Y - C[1])
    tt = np.arctan2(Y - C[1], X - C[0])
    phase = rr / 6.5 + tt / (2 * np.pi)
    bandw = np.floor(phase).astype(int)
    cut = np.floor((np.degrees(tt) + 180 + 23 * bandw) / 62.0).astype(int)
    M.paint_split(inw, bandw * 100 + cut, "wave")
    # the tube: the hollow under the lip
    crest_x = P[np.argmax(P[:, 1]), 0]
    tube = rr < room * 1.08
    M.paint_split(tube, (rr > room * 0.7).astype(int), "tube")
    # white water: a scalloped band along the lip and the curl
    scal = 4.2 + 1.3 * np.sin(tt * 7 + rr / 3.0)
    foam = inw & (-dw < scal) & (X < crest_x + 6) & (Y > WAVE["foot"] + 14)
    M.paint_split(foam, np.floor((np.degrees(tt) + 180) / 40.0).astype(int), "foam")
    # whitewater where the wave meets the sea
    sea_line = WAVE["foot"] + 3 + 1.2 * np.sin(X / 4.0)
    wash = (Y < sea_line) & (Y > WAVE["foot"] - 4 - 1.5 * np.sin(X / 5.5 + 1)) & (X < crest_x + 14) & (X > -64)
    M.paint_split(wash, np.floor((X + 2 * np.sin(Y)) / 13.0).astype(int), "foam")

    # ti leaves along the bottom, behind the flowers
    leaves = [((-56, -48), 50, 38, 14, 0.12), ((-30, -70), 95, 34, 13, -0.12),
              ((-8, -72), 70, 34, 12, 0.12), ((22, -72), 105, 32, 12, -0.10),
              ((58, -46), 125, 36, 14, -0.12), ((64, -30), 160, 22, 10, -0.12)]
    for base, ang, L, Wd, bend in leaves:
        for k, h in enumerate(leaf_poly(np.array(base, float), np.radians(ang), L, Wd, bend)):
            M.paint(nalu.poly_sdf(X, Y, h) < 0, "leaf")

    # flowers
    def plumeria(Cf, R, rot, cat="plumeria"):
        for k in range(5):
            Pp = petal_poly(R, 2 * np.pi * k / 5 + rot) + Cf
            M.paint(nalu.poly_sdf(X, Y, Pp) < 0, cat)
        M.paint(np.hypot(X - Cf[0], Y - Cf[1]) < 0.24 * R, "plum_c")

    def hibiscus(Cf, R, rot, cat="hibiscus"):
        for k in range(5):
            Pp = petal_poly(R, 2 * np.pi * k / 5 + rot, ruffle=0.05, width=0.62) + Cf
            M.paint(nalu.poly_sdf(X, Y, Pp) < 0, cat)
        M.paint(np.hypot(X - Cf[0], Y - Cf[1]) < 0.22 * R, "hib_c")

    hib = (np.array([27.0, -45.0]), 17.0, 0.5)
    hibiscus(*hib)
    plumeria(np.array([-24.0, -47.0]), 14.0, 0.2)
    plumeria(np.array([2.0, -58.0]), 10.0, 1.1, cat="pink")
    plumeria(np.array([-46.0, -35.0]), 9.0, 0.7)

    # free lines: the hibiscus stamen with its pollen dots
    Hc, HR, _ = hib
    a = np.radians(62)
    tip = Hc + 0.95 * HR * np.array([np.cos(a), np.sin(a)])
    dfree = seg_dist(X, Y, Hc + 0.22 * HR * np.array([np.cos(a), np.sin(a)]), tip)
    for k in range(5):
        b = a + np.radians(-50 + 25 * k)
        q = tip + 2.6 * np.array([np.cos(b), np.sin(b)])
        dfree = np.minimum(dfree, np.hypot(X - q[0], Y - q[1]) - 0.5)

    # outside the frame opening
    M.lab[r > R_IN + 0.4] = 0
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
