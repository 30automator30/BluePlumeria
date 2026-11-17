#!/usr/bin/env python3
"""Stained-glass suncatcher engine (clear resin, then hand-painted).

A suncatcher is a flat clear pane crossed by raised "lead" lines, in a round
frame with a hanging loop. Every piece between the lines is closed, so each
holds its own glass paint like a little dam.

A design is a small scene module (see ../wave-suncatcher/suncatcher.py)
that defines:
  FRAME    a Frame (size, pane, lines, loop, merge limits)
  PALETTE  {category: [rgb, ...]}, the suggested paint colours
  draw(M)  paints the scene onto the Map M: M.X, M.Y are the grid (mm,
           origin at the centre, y up), M.F the frame. Use M.paint(mask,
           cat), M.paint_split(mask, keys, cat), shatter(...) for
           hand-cut pieces, and M.line(dist) for free lines (veins, stamens).
           Later paint goes on top.
  OUT      the STL file name
and ends with  `if __name__ == "__main__": stained_glass.main(sys.modules[__name__])`.

The engine then splits the painted regions into connected pieces, merges
pieces too small or thin to paint into a neighbour, runs a lead line along
every boundary, and builds the solid: pane (flat back), lines (rounded tops,
straight sides), a rounded frame ring and a hanging loop. It also draws the
paint guide (a painted mock-up beside the line drawing).

The cut pattern depends on the pitch it is drawn at, so it is always drawn
at FRAME.design_res (0.2 mm); the final mesh resamples the smoothed line
field onto a finer grid, so the final keeps the approved pattern.

Scene CLI (provided by main):
  python3 scene.py --guide-only                 # paint guide only (fast)
  python3 scene.py --voxel 0.2 --preview        # quick mesh + guide
  python3 scene.py --voxel 0.1                  # final (decimated)
"""

import argparse
import dataclasses
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("fractal-plumeria", "plumeria-ring"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
from fractal_plumeria import mesh_volume  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402


@dataclasses.dataclass
class Frame:
    d: float = 150.0          # outer diameter, mm
    ring_w: float = 5.0       # frame ring width
    ring_t: float = 4.0       # frame ring thickness
    pane_t: float = 1.6       # clear pane thickness (back is flat)
    line_w: float = 1.0       # lead line width
    line_h: float = 1.0       # lead line height above the pane
    loop_r: float = 4.0       # hanging loop hole radius
    loop_tube: float = 2.0    # hanging loop tube radius
    min_area: float = 5.0     # mm^2: smaller pieces are merged
    min_half: float = 1.15    # mm: pieces thinner than 2x this are merged
    design_res: float = 0.2   # pitch the design is drawn at; keep fixed once approved

    @property
    def r_out(self):
        return self.d / 2

    @property
    def r_in(self):
        return self.d / 2 - self.ring_w


def poly_sdf(px, py, P):
    """Signed distance in 2D from points to a closed polygon P (m, 2)."""
    a, b = P[:-1], P[1:]
    best = np.full(px.shape, 1e9)
    inside = np.zeros(px.shape, bool)
    for (ax, ay), (bx, by) in zip(a, b):
        ex, ey = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        h = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey + 1e-12), 0, 1)
        best = np.minimum(best, (wx - h * ex) ** 2 + (wy - h * ey) ** 2)
        c = ((ay <= py) & (by > py)) | ((by <= py) & (ay > py))
        xint = ax + (py - ay) * ex / (ey + 1e-12)
        inside ^= c & (px < xint)
    d = np.sqrt(best)
    return np.where(inside, -d, d)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def smax(a, b, k):
    return -smin(-a, -b, k)


# ---------------------------------------------------------------- shapes --

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

    def __init__(self, X, Y, frame):
        self.X, self.Y = X, Y
        self.F = frame
        self.r = np.hypot(X, Y)
        self.lab = np.zeros(X.shape, np.int32)
        self.cat = ["none"]
        self.dfree = np.full(X.shape, 1e9)       # distance to the free lines

    def line(self, d):
        """Add a free lead line (one that may end inside a piece): d is the
        distance to its centre line (minus any radius, for dots)."""
        self.dfree = np.minimum(self.dfree, d)

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


# ------------------------------------------------- ready-made subjects --

def hibiscus(M, C, R, rot, cat="hibiscus", eye="hib_c"):
    """A hibiscus: five broad frilled petals (each with a dark eye piece at
    the centre and a vein), a centre, and a curved staminal column with
    pollen dots."""
    X, Y = M.X, M.Y
    C = np.array(C, float)
    for k in range(5):
        a = rot + 2 * np.pi * k / 5
        P = hib_petal(C, R, a)
        m = poly_sdf(X, Y, P) < 0
        rr = np.hypot(X - C[0], Y - C[1])
        u, v = local(X, Y, C, a)
        inner = rr < R * (0.36 + 0.06 * np.sin(np.arctan2(v, u) * 9))
        M.paint(m & ~inner, cat)
        M.paint(m & inner, eye)
        M.line(seg_dist(X, Y, C + 0.42 * R * np.array([np.cos(a), np.sin(a)]),
                        C + 0.74 * R * np.array([np.cos(a), np.sin(a)])))
    M.paint(np.hypot(X - C[0], Y - C[1]) < 0.12 * R, eye)
    sa = rot + np.radians(20)
    t = np.linspace(0.12, 1.05, 12)
    pts = C + R * np.stack([t * np.cos(sa + 0.25 * t), t * np.sin(sa + 0.25 * t)], 1)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        M.line(seg_dist(X, Y, p0, p1))
    tip = pts[-1]
    for k in range(6):
        b = sa + np.radians(-75 + 30 * k)
        q = tip + 0.12 * R * np.array([np.cos(b), np.sin(b)])
        M.line(np.hypot(X - q[0], Y - q[1]) - 0.55)


def plumeria(M, C, R, rot, cat="plumeria", throat="plum_c"):
    """A plumeria: five pinwheel petals, each with a throat piece, and a
    small centre."""
    X, Y = M.X, M.Y
    C = np.array(C, float)
    for k in range(5):
        a = 2 * np.pi * k / 5 + rot
        P = petal_poly(R, a, width=0.47) + C
        m = poly_sdf(X, Y, P) < 0
        off = 0.15 * R
        c = C + np.array([-off * np.sin(a), off * np.cos(a)])
        u, v = local(X, Y, c, a)
        th = np.hypot(X - C[0], Y - C[1]) < 0.40 * R * (1 + 0.12 * np.sin(4 * np.arctan2(v, u)))
        M.paint(m & ~th, cat)
        M.paint(m & th, throat)
    M.paint(np.hypot(X - C[0], Y - C[1]) < 0.13 * R, throat)


def leaf(M, base, ang_deg, L, W, bend=0.1, vein=5.0, cat="leaf", lean=0.9, clip=None):
    """A long pointed leaf split by its midrib and slanting side veins every
    `vein` mm."""
    X, Y = M.X, M.Y
    a = np.radians(ang_deg)
    base = np.array(base, float)
    u, v = local(X, Y, base, a)
    for h in leaf_poly(base, a, L, W, bend):
        m = poly_sdf(X, Y, h) < 0
        if clip is not None:
            m &= clip
        M.paint_split(m, np.floor((u - lean * np.abs(v)) / vein).astype(int), cat)


# ---------------------------------------------------------------- design --

def design(scene, res):
    """Draw the scene on a grid of pitch res. Returns xs, the region map
    (0 = outside the pane), each region's category, and the distance to the
    free lines."""
    F = scene.FRAME
    xs = np.arange(-F.r_in - 1, F.r_in + 1 + res, res)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    M = Map(X, Y, F)
    scene.draw(M)
    M.lab[M.r > F.r_in + 0.4] = 0
    M.dfree[M.r > F.r_in] = 1e9
    return xs, M.lab, M.cat, M.dfree


def seg_dist(X, Y, a, b):
    e = b - a
    h = np.clip(((X - a[0]) * e[0] + (Y - a[1]) * e[1]) / (e @ e), 0, 1)
    return np.hypot(X - a[0] - h * e[0], Y - a[1] - h * e[1])


# ------------------------------------------------------------- the cells --

def cells(lab, res, F):
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
        bad = [i for i, a in zip(ids, area) if i > 0 and (a < F.min_area or half[i] < F.min_half)]
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

def build(scene, voxel):
    t0 = time.time()
    F = scene.FRAME
    dr = F.design_res
    xs, lab, cat, dfree = design(scene, dr)
    cid, src = cells(lab, dr, F)
    ncell = cid.max()
    print(f"  {ncell} cells, design {time.time() - t0:.0f}s")
    dline = line_distance(cid, dfree, dr)
    # soften the pixel steps of the boundaries before resampling
    dsmooth = ndi.gaussian_filter(dline, 1.0)
    # extend the 2D fields to the full grid (frame and loop above it)
    lo = np.array([-F.r_out - 1.0, -F.r_out - 1.0, -0.4])
    ys_top = F.r_out + 2 * F.loop_r + 2 * F.loop_tube + 2
    gx = np.arange(lo[0], F.r_out + 1.0, voxel)
    gy = np.arange(lo[1], ys_top, voxel)
    gz = np.arange(lo[2], F.ring_t + 0.4, voxel)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    r = np.hypot(X, Y)
    # resample the line distance onto the mesh grid (bilinear)
    fi = (X - xs[0]) / dr
    fj = (Y - xs[0]) / dr
    DL = ndi.map_coordinates(dsmooth, [fi.ravel(), fj.ravel()], order=1, mode="nearest").reshape(X.shape)
    hw, rr = F.line_w / 2, 0.45                       # half-width, top rounding
    top = F.pane_t + F.line_h
    open_ = F.r_in + 0.5 - r                          # >0 inside the frame's opening
    ring_r = np.abs(r - (F.r_out - F.ring_w / 2)) - F.ring_w / 2
    lc = np.array([0.0, F.r_out + F.loop_r + F.loop_tube * 0.3, F.ring_t / 2])
    q = np.hypot(X - lc[0], Y - lc[1]) - F.loop_r - F.loop_tube
    vol = np.empty((len(gx), len(gy), len(gz)), np.float32)
    for k, z in enumerate(gz):
        pane = np.maximum(np.maximum(z - F.pane_t, -z), -open_)
        # a line: a rounded box in (distance across, height)
        qx, qz = DL - (hw - rr), z - (top - rr)
        line = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - rr
        line = np.maximum(np.maximum(line, -z), -open_)
        frame = smax(ring_r, np.abs(z - F.ring_t / 2) - F.ring_t / 2, 1.2)
        d = np.minimum(np.minimum(pane, line), frame)
        d = smin(d, np.hypot(q, z - lc[2]) - F.loop_tube, 0.8)
        vol[:, :, k] = d
    print(f"  built {time.time() - t0:.0f}s")
    return vol, (gx[0], gy[0], gz[0]), (xs, cid, src, cat, dline)


def guide(scene, xs, cid, src, cat, dline, out, numbered=True):
    """Paint guide: the cells coloured as suggested, lines dark."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F, PALETTE = scene.FRAME, scene.PALETTE
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
    hw = F.line_w / 2
    img[dline < hw] = (0.18, 0.18, 0.2)
    r = np.hypot(*np.meshgrid(xs, xs, indexing="ij"))
    img[(r > F.r_in) & (r < F.r_out)] = (0.18, 0.18, 0.2)
    img = np.transpose(img, (1, 0, 2))[::-1]
    fig, ax = plt.subplots(1, 2 if numbered else 1, figsize=(16 if numbered else 8, 8), squeeze=False)
    ax = ax.ravel()
    ext = (xs[0], xs[-1], xs[0], xs[-1])
    ax[0].imshow(img, extent=ext)
    ax[0].set_title("painted (suggested colours)", fontsize=13)
    if numbered:
        line = np.ones(cid.shape + (3,))
        line[dline < hw] = 0.2
        line[(r > F.r_in) & (r < F.r_out)] = 0.2
        ax[1].imshow(np.transpose(line, (1, 0, 2))[::-1], extent=ext)
        ax[1].set_title("lead lines (as printed)", fontsize=13)
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def main(scene):
    ap = argparse.ArgumentParser(description=scene.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.2)
    ap.add_argument("--preview", action="store_true", help="skip decimation")
    ap.add_argument("--guide-only", action="store_true", help="paint guide only")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(scene.__file__)))
    args = ap.parse_args()
    F = scene.FRAME
    t0 = time.time()
    guide_out = os.path.join(args.outdir, "paint_guide.png")
    if args.guide_only:
        xs, lab, cat, dfree = design(scene, F.design_res)
        cid, src = cells(lab, F.design_res, F)
        print(f"{cid.max()} pieces")
        guide(scene, xs, cid, src, cat, line_distance(cid, dfree, F.design_res), guide_out)
        return
    vol, origin, g = build(scene, args.voxel)
    mesh = mesh_volume(vol, args.voxel, origin)
    del vol
    if not args.preview:
        mesh = decimate(mesh, 0.004)
    print(f"suncatcher: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, {time.time() - t0:.0f}s")
    out = os.path.join(args.outdir, scene.OUT)
    mesh.export(out)
    guide(scene, *g, guide_out)
    print("wrote", out)
