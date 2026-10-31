#!/usr/bin/env python3
"""Plumeria Echo series: strung focal pieces for Amy's beaded necklaces.

One hero bloom plus four smaller "echoes" (bud, single petal, half bloom,
leaf). Every piece is a solid relief built around the same strung spine: a
channel running side to side through a solid core, flared at both ends, with
a shallow cupped seat at each end so neighbouring beads nestle in.

Axes (mm): strand along X, Y is up when worn, the face looks toward +Z and
the back lies flat at Z = 0 (the bud is round, like a bead).

Usage
  python echo_series.py --preview          # renders only (for approval)
  python echo_series.py --stl              # write STLs after approval
  python echo_series.py --only hero --voxel 0.04 --stl
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plumeria-ring"))
from plumeria_ring import decimate, petal_shape, smin  # noqa: E402

# ---------------------------------------------------------------- the spine
CH_R = 0.8          # strand channel radius (1.6 mm: beading wire, silk, doubled thread)
CH_FLARE = 0.45     # extra radius at each mouth so wire never rubs a sharp edge
CORE_R = 2.0        # solid core radius -> 1.2 mm of resin around the channel
CORE_Z = 2.0        # channel height above the flat back
SEAT_R, SEAT_D = 3.2, 0.7   # cupped bead seat: sphere radius, depth
COLLAR = 0.8        # how far the core ends stand proud of the outline


def smax(a, b, k):
    return -smin(-a, -b, k)


def spine(x, y, z, half_len, zc=CORE_Z):
    """Solid core along X with the channel, flared mouths and bead seats cut in."""
    ax = np.abs(x)
    core = np.sqrt(np.maximum(ax - half_len, 0) ** 2 + y * y + (z - zc) ** 2) - CORE_R
    end = half_len + CORE_R
    r = CH_R + CH_FLARE * np.clip((ax - (end - 1.0)) / 1.0, 0, 1) ** 2
    channel = np.sqrt(y * y + (z - zc) ** 2) - r
    seat = np.sqrt((ax - (end + SEAT_R - SEAT_D)) ** 2 + y * y + (z - zc) ** 2) - SEAT_R
    return core, np.minimum(channel, seat)


# ---------------------------------------------------------------- motifs
# Petals and leaves are curved shells (1.2-2 mm) that sit on top of the strand
# tube, like a flower soldered to a tube bail: robust, but they still read as
# real petals instead of a slab.
SHELL_BASE = CORE_Z + CORE_R - 0.6     # petal mid-surface where it meets the tube


# Epoxy beds: every petal, leaf and makau is a shallow well with a raised rail
# around it, ready for an epoxy / UV-resin overlay. The bed depth is added on
# top of each shell, so the floor under the epoxy keeps the full shell thickness.
BED, RAIL = 0.7, 0.65      # well depth, rail width (mm)


def petal(x, y, z, cx, cy, ang, L, wmax, lift=1.0, tb=2.0, tt=1.3, twist=0.6,
          cup=0.5, curl=0.8, base=SHELL_BASE, bed=BED, rail=RAIL):
    """One plumeria petal: obovate outline, rising from the throat, twisted so
    neighbours overlap, cupped, tip curling up; a raised rail runs round the
    edge and encloses a bed for epoxy."""
    dx, dy = np.cos(ang), np.sin(ang)
    px, py = x - cx, y - cy
    u = px * dx + py * dy
    v = -px * dy + py * dx
    t = np.clip(u / L, 0, 1)
    w = wmax * petal_shape(t) * np.where(v > 0, 1.12, 0.88) + 1e-3
    outline = np.maximum(np.abs(v) - w, np.maximum(-u, u - L))
    vn = np.clip(v / w, -1, 1)
    tip = np.maximum(t - 0.6, 0) / 0.4
    mid = (base + lift * np.sin(np.pi * 0.5 * np.minimum(t * 1.6, 1))
           + twist * vn * t + cup * vn * vn * t + curl * tip * tip)
    thick = tt + (tb - tt) * (1 - t)
    floor = mid + thick / 2
    shell = np.abs(z - (mid + bed / 2)) / 1.5 - (thick + bed) / 2
    d = smax(outline, shell, 0.3)
    well = np.maximum(np.maximum(outline + rail, 0.2 * L - u), (floor - z) / 1.5)
    return smax(d, -well, 0.15)


def throat(x, y, z, r, top):
    """Flower centre: a rounded dome with a small funnel."""
    rr = np.sqrt(x * x + y * y)
    dome = np.sqrt(x * x + y * y + (np.maximum(z - top, 0) * 1.6) ** 2) - r
    dome = np.maximum(dome, CORE_Z - z)
    funnel = np.sqrt(rr * rr + (z - top - 0.75 * r) ** 2) - 0.32 * r
    return smax(dome, -funnel, 0.15)


def bloom(x, y, z, R, angles, **kw):
    d = np.full(x.shape, 1e9, np.float32)
    for a in angles:
        off = 0.14 * R                                  # pinwheel offset
        cx, cy = -off * np.sin(a), off * np.cos(a)
        d = np.minimum(d, petal(x, y, z, cx, cy, a, R, 0.40 * R, **kw))
    return smin(d, throat(x, y, z, 0.2 * R, SHELL_BASE + 0.6 + BED), 0.4)


def leaf_blade(x, y, z, L, hw, lift=0.8, tb=1.5, te=1.05, base=SHELL_BASE,
               sweep=0.16, sway=0.05, twist=0.45, curl=0.9, bed=BED, rail=RAIL):
    """Plumeria leaf along X, drawn as if moving: the midline sweeps to one side
    (sweep) with a soft S (sway), the blade twists along its length, the edge
    ripples gently and the tip lifts. A rail runs round the edge and down the
    midrib, enclosing two epoxy beds; side veins are engraved in the bed floor."""
    u = x + L / 2
    t = np.clip(u / L, 0, 1)
    y = y - L * (sweep * t * t + sway * np.sin(2 * np.pi * t))    # curved midline
    ripple = 1 + 0.07 * np.sin(5.5 * np.pi * t) * np.sin(np.pi * t)
    w = hw * ripple * np.maximum(np.sin(np.pi * t ** 0.72), 0) ** 0.9 * (1.08 - 0.25 * t) + 1e-3
    outline = np.maximum(np.abs(y) - w, np.maximum(-u, u - L))
    yn = np.clip(np.abs(y) / w, 0, 1)
    ys = np.clip(y / w, -1, 1)
    tip = np.maximum(t - 0.65, 0) / 0.35
    mid = (base + lift * np.maximum(np.sin(np.pi * t), 0) ** 0.6 - 0.6 * yn ** 2 + 0.25 * t
           + twist * ys * np.sin(np.pi * t) + curl * tip * tip)
    thick = te + (tb - te) * (1 - yn)
    floor = mid + thick / 2
    d = smax(outline, np.abs(z - (mid + bed / 2)) / 1.35 - (thick + bed) / 2, 0.3)
    well = np.maximum(np.maximum(outline + rail, rail / 2 - np.abs(y)), (floor - z) / 1.35)
    d = smax(d, -well, 0.12)
    vein = np.abs(np.sin(np.pi * (u - 1.4 * np.abs(y)) / 2.1)) * 2.1 / np.pi - 0.15
    vein = np.maximum(vein, np.maximum(rail / 2 + 0.25 - np.abs(y), np.abs(y) - (w - rail - 0.2)))
    vein = np.maximum(vein, (floor - 0.25) - z)
    return np.maximum(d, -vein) / 1.15


def bud_body(x, y, z, L, rmax, zc):
    """Twisted closed plumeria bud lying along the strand, like a carved bead."""
    t = np.clip((x + L / 2) / L, 0, 1)
    prof = CORE_R + (rmax - CORE_R) * np.maximum(np.sin(np.pi * t ** 0.7), 0) ** 0.9
    th = np.arctan2(z - zc, y)
    rr = np.sqrt(y * y + (z - zc) ** 2)
    spiral = 1 + 0.16 * np.cos(5 * th + 3.0 * np.pi * t) * np.maximum(np.sin(np.pi * t), 0) ** 0.7
    d = (rr - prof * spiral) * 0.75
    return np.maximum(d, np.abs(x) - L / 2)


def local(x, y, cx, cy, ang):
    """Coordinates in a frame at (cx, cy) rotated by ang (its +x along ang)."""
    c, s_ = np.cos(ang), np.sin(ang)
    px, py = x - cx, y - cy
    return px * c + py * s_, -px * s_ + py * c


def polyline_dist(x, y, pts, hw):
    """Distance from (x, y) to a polyline, and the interpolated half-width there."""
    best = np.full(x.shape, 1e9, np.float32)
    w = np.zeros(x.shape, np.float32)
    for i in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        ex, ey = bx - ax, by - ay
        h = np.clip(((x - ax) * ex + (y - ay) * ey) / (ex * ex + ey * ey), 0, 1)
        d = np.hypot(x - ax - h * ex, y - ay - h * ey)
        wi = hw[i] + (hw[i + 1] - hw[i]) * h
        closer = d - wi < best - w
        best, w = np.where(closer, d, best), np.where(closer, wi, w)
    return best, w


# The makau outline is traced from Amy's reference photo (makau_outline.npz:
# hook mask and lashing band, in image pixels; point on the left, shank on the
# right). Built here as a plate with a raised rim around a recessed inlay
# panel, raised grooved lashing, and every tip rounded so nothing can snag.
_MAKAU = {}


def _makau_fields(height):
    """Signed 2D distance fields (mm) for the hook, cached per height."""
    if height not in _MAKAU:
        from scipy import ndimage
        o = np.load(os.path.join(HERE, "makau_outline.npz"))
        hook, lash = o["hook"], o["lash"]
        mm = height / hook.shape[0]                       # mm per pixel
        tip_r = max(1, int(round(0.35 / mm)))             # round every tip to 0.35 mm
        er = ndimage.distance_transform_edt(hook) > tip_r
        sd = (ndimage.distance_transform_edt(~er) - ndimage.distance_transform_edt(er)) * mm - tip_r * mm
        lash_sd = (ndimage.distance_transform_edt(~lash) - ndimage.distance_transform_edt(lash)) * mm
        ly, lx = np.nonzero(lash)
        _MAKAU[height] = (sd.astype(np.float32), lash_sd.astype(np.float32), mm,
                          (lx.mean(), ly.mean()))
    return _MAKAU[height]


def _sample(field, px, py):
    from scipy import ndimage
    return ndimage.map_coordinates(field, [py.ravel(), px.ravel()], order=1,
                                   mode="constant", cval=5.0).reshape(px.shape)


def makau(x, y, z, cx, cy, mirror=False, height=16.0, zm=1.55, T=3.0, rim=RAIL, depth=0.75):
    """Makau plate whose lashing centre sits at (cx, cy). Left-side hook (point
    outward on the left); mirror=True for the right side."""
    sd, lash_sd, mm, (ax, ay) = _makau_fields(height)
    lx = -(x - cx) if mirror else (x - cx)
    px, py = ax + lx / mm, ay - (y - cy) / mm
    d2 = _sample(sd, px, py)
    d = smax(d2, np.abs(z - zm) - T / 2, 0.6)
    lash2 = _sample(lash_sd, px, py)
    inlay = np.maximum(d2 + rim, (zm + T / 2 - depth) - z)
    inlay = np.maximum(inlay, -(lash2 - 0.6))                     # keep the panel off the lashing
    d = np.maximum(d, -inlay)
    # lashing: raised wraps with grooves between them
    wrap = smax(np.maximum(lash2, d2 - 0.25), np.abs(z - zm) - (T / 2 + 0.35), 0.2)
    groove = np.abs(np.sin(np.pi * (py - ay) * mm / 0.62)) * 0.62 / np.pi - 0.07
    d = np.minimum(d, np.maximum(wrap, -groove))
    return np.maximum(d, -z)


def leaf_at(x, y, z, cx, cy, ang, L, hw, base, flip=False, **kw):
    lx, ly = local(x, y, cx, cy, ang)
    if flip:
        ly = -ly
    return leaf_blade(lx, ly, z, L, hw, base=base, **kw)


# ---------------------------------------------------------------- the series
class Piece:
    def __init__(self, name, title, bounds, sdf, axis=(0.0, CORE_Z), bead=None):
        self.name, self.title, self.bounds, self.sdf = name, title, bounds, sdf
        self.axis = axis          # (y, z) of the strand line
        self.bead = bead          # (x, radius) of a bead threaded through the piece


def with_spine(body, half_len, zc=CORE_Z):
    def f(x, y, z):
        core, cut = spine(x, y, z, half_len, zc)
        return np.maximum(smin(body(x, y, z), core, 0.6), -cut)
    return f


# Hero composition (mm). The strand runs through tubes behind the two side
# blooms; the centre bead threads onto the wire in the gap between them, and
# the necklace carries on from the outer seats as more beads or chain.
BEAD_GAP = 10.0            # inner seat to inner seat: a 10 mm bead, or a smaller bead + spacers
HERO_STRAND_Y = 4.5
HERO_TUBE_HALF = 5.0
HERO_TUBE_X = BEAD_GAP / 2 + CORE_R + HERO_TUBE_HALF


def hero_sdf(x, y, z):
    up = np.pi / 2
    ys, side_cx = HERO_STRAND_Y, HERO_TUBE_X + 1.5
    # layer 1: plumerias. The big bloom has a notch at the top for the bead.
    big = [-up + 2 * np.pi * i / 5 for i in range(5)]
    d = bloom(x, y + 12.0, z - 0.4, 11.0, big)
    left = [-up + 0.18 + 2 * np.pi * i / 5 for i in range(5)]       # side blooms turned a touch
    right = [-up - 0.30 + 2 * np.pi * i / 5 for i in range(5)]      # ...and not quite mirrored
    d = np.minimum(d, bloom(x + side_cx, y - ys + 0.3, z, 7.5, left,
                            tb=1.8, tt=1.25, lift=0.85, curl=0.7))
    d = np.minimum(d, bloom(x - side_cx, y - ys + 0.8, z, 7.2, right,
                            tb=1.8, tt=1.25, lift=0.85, curl=0.7))
    # layer 2: leaves sweeping out from behind the big bloom, under the side blooms
    lb = SHELL_BASE - 1.1
    d = np.minimum(d, leaf_at(x, y, z, -12.0, -5.6, np.pi - 0.10, 21.0, 3.9, lb, sweep=-0.2, sway=0.03))
    d = np.minimum(d, leaf_at(x, y, z, 12.5, -5.3, 0.06, 22.0, 3.9, lb, flip=True, sweep=-0.19, sway=-0.03))
    # layer 3: makau at the outer ends, shank under the leaf tip and side bloom
    hx = HERO_TUBE_X + 7.0
    d = np.minimum(d, makau(x, y, z, -hx, -3.0))
    d = np.minimum(d, makau(x, y, z, hx + 0.4, -2.5, mirror=True, height=16.8))
    for sx in (-HERO_TUBE_X, HERO_TUBE_X):                          # strand tubes + seats
        core, cut = spine(x - sx, y - ys, z, HERO_TUBE_HALF)
        d = np.maximum(smin(d, core, 0.6), -cut)
    return d


def secondary_sdf(x, y, z):
    """Secondary hero (left-hand; the right one is its mirror): the same three
    layers, calmer: one bloom, one leaf trailing outward, one small makau."""
    up = np.pi / 2
    d = bloom(x - 1.5, y + 1.0, z, 8.0, [-up + 0.25 + 2 * np.pi * i / 5 for i in range(5)],
              tb=1.8, tt=1.25, lift=0.85, curl=0.7)
    d = np.minimum(d, leaf_at(x, y, z, -8.0, -4.0, np.pi + 0.55, 15.0, 3.2, SHELL_BASE - 1.1))
    d = np.minimum(d, makau(x, y, z, 7.5, -3.0, mirror=True, height=12.0))
    core, cut = spine(x, y, z, 6.0)
    return np.maximum(smin(d, core, 0.6), -cut)


def make_series():
    up = np.pi / 2
    bloom_angles = [-up + 2 * np.pi * i / 5 for i in range(5)]      # one petal points down
    hook_sdf = with_spine(lambda x, y, z: makau(x, y, z, 0.0, -3.4, height=14.0), 4.2)
    hx = HERO_TUBE_X + 7.4
    return [
        Piece("hero", "Hero: three blooms, leaves, makau (bead threads in the gap)",
              ((-hx - 11.5, hx + 11.5), (-25.0, 14.0), (0, 9)), hero_sdf,
              axis=(HERO_STRAND_Y, CORE_Z), bead=(0.0, BEAD_GAP / 2)),
        Piece("secondary", "Secondary hero, left (mirror for the right)",
              ((-13.5, 15.5), (-15.5, 10.0), (0, 8.5)), secondary_sdf),
        Piece("bloom", "Bloom (18 mm)", ((-11.5, 11.5), (-10.5, 10.5), (0, 8.5)),
              with_spine(lambda x, y, z: bloom(x, y, z, 9.0, bloom_angles,
                                               tb=1.8, tt=1.25, lift=0.9, curl=0.7), 7.6)),
        Piece("half_bloom", "Half bloom (15 mm)", ((-9.5, 9.5), (-3.5, 10), (0, 8)),
              with_spine(lambda x, y, z: bloom(x, y, z, 8.0, [up - 0.7, up + 0.55],
                                               tb=1.7, tt=1.2, lift=0.8, curl=0.7), 4.0)),
        Piece("hook", "Makau (14 mm)", ((-9, 9), (-15, 3.5), (0, 6)), hook_sdf),
        Piece("leaf", "Leaf (16 mm)", ((-10.5, 10.5), (-5.5, 8.0), (0, 8)),
              with_spine(lambda x, y, z: leaf_blade(x, y, z, 16.0, 3.8), 6.5)),
        Piece("bud", "Bud (14 mm)", ((-9, 9), (-4.5, 4.5), (0, 8)),
              with_spine(lambda x, y, z: bud_body(x, y, z, 14.0, 3.4, 3.4), 5.0, zc=3.4),
              axis=(0.0, 3.4)),
    ]


# ---------------------------------------------------------------- meshing
def mesh_sdf(piece, voxel):
    from skimage.measure import marching_cubes
    import trimesh
    (x0, x1), (y0, y1), (z0, z1) = piece.bounds
    xs = np.arange(x0, x1 + voxel, voxel, dtype=np.float32)
    ys = np.arange(y0, y1 + voxel, voxel, dtype=np.float32)
    zs = np.arange(z0 - 0.3, z1 + voxel, voxel, dtype=np.float32)
    gy, gz = np.meshgrid(ys, zs, indexing="ij")
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    for i, xv in enumerate(xs):
        vol[i] = piece.sdf(np.full_like(gy, xv), gy, gz)
    from scipy import ndimage
    _, bodies = ndimage.label(vol < 0, structure=np.ones((3, 3, 3)))
    vol = np.pad(vol, 1, constant_values=1.0)
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(voxel,) * 3)
    verts += np.array([xs[0], ys[0], zs[0]]) - voxel
    mesh = trimesh.Trimesh(verts, faces, process=True)
    parts = mesh.split(only_watertight=False)          # drop zero-area meshing specks
    return max(parts, key=lambda m: len(m.faces)), bodies


def check(piece, mesh, nparts):
    """Watertight, one body, and the strand channel is open end to end."""
    yc, zc = piece.axis
    (x0, x1) = piece.bounds[0]
    # open channel: a 1.2 mm wire along the axis never touches resin
    xs = np.linspace(x0, x1, 4000, dtype=np.float32)
    ok_channel = True
    for dy, dz in [(0, 0), (0.6, 0), (-0.6, 0), (0, 0.6), (0, -0.6)]:
        d = piece.sdf(xs, np.full_like(xs, yc + dy), np.full_like(xs, zc + dz))
        ok_channel &= bool((d > 0).all())
    ok_bead = True
    if piece.bead:                      # the threaded bead must not touch resin
        bx, br = piece.bead
        u = np.random.default_rng(0).normal(size=(20000, 3)).astype(np.float32)
        u = u / np.linalg.norm(u, axis=1, keepdims=True) * br * np.cbrt(
            np.random.default_rng(1).random((20000, 1)).astype(np.float32))
        ok_bead = bool((piece.sdf(bx + u[:, 0], yc + u[:, 1], zc + u[:, 2]) > 0).all())
    print(f"  {piece.name}: watertight={mesh.is_watertight} bodies={nparts} "
          f"channel_open={ok_channel} bead_clear={ok_bead} size={np.round(mesh.extents, 1)} mm "
          f"vol={mesh.volume / 1000:.2f} cm3 tris={len(mesh.faces):,}")
    return mesh.is_watertight and ok_channel and ok_bead and nparts == 1


# ---------------------------------------------------------------- previews
def rot(ax, a):
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[ax]


def splat(meshes, R, span, px=600, center=(0, 0)):
    """Point-splat z-buffer render (no GPU): list of (mesh, offset, colour)."""
    import trimesh
    from scipy.ndimage import grey_dilation
    img = np.ones((px, px, 3)); depth = np.full((px, px), -1e9)
    L = np.array([-0.4, 0.5, 0.77]); L /= np.linalg.norm(L)
    for mesh, off, col in meshes:
        pts, fi = trimesh.sample.sample_surface(mesh, 1_500_000)
        p = (pts + off) @ R.T
        n = mesh.face_normals[fi] @ R.T
        ix = (((p[:, 0] - center[0]) / span + 0.5) * px).astype(int)
        iy = ((0.5 - (p[:, 1] - center[1]) / span) * px).astype(int)
        ok = (ix >= 0) & (ix < px) & (iy >= 0) & (iy < px)
        ix, iy, p, n = ix[ok], iy[ok], p[ok], n[ok]
        lam = np.clip(n @ L, 0, 1)
        c = np.clip((0.25 + 0.7 * lam)[:, None] * np.array(col) + 0.35 * (lam ** 30)[:, None], 0, 1)
        o = np.argsort(p[:, 2])
        ix, iy, z, c = ix[o], iy[o], p[o, 2], c[o]
        win = z > depth[iy, ix]
        img[iy[win], ix[win]] = c[win]; depth[iy[win], ix[win]] = z[win]
    holes = depth < -1e8
    near = grey_dilation(np.where(holes, -1e9, depth), size=3) > -1e8
    for ch in range(3):
        dil = grey_dilation(np.where(holes, 0, img[..., ch]), size=2)
        img[..., ch] = np.where(holes & near, dil, img[..., ch])
    return img


RESIN = (0.80, 0.80, 0.88)


BEAD_COL = (0.25, 0.35, 0.75)


def bead_items(piece, offset):
    """Placeholder bead threaded through a piece (the hero's centre bead)."""
    import trimesh
    if not piece.bead:
        return []
    bx, br = piece.bead
    yc, zc = piece.axis
    return [(trimesh.creation.icosphere(4, radius=br), np.array([bx, yc, zc]) + offset, BEAD_COL)]


def render_pieces(meshes, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    views = [("Front", np.eye(3)),
             ("3/4", rot("x", -0.55) @ rot("y", 0.6)),
             ("Strand end (channel + seat)", rot("y", -np.pi / 2)),
             ("Back", rot("y", np.pi))]
    fig, axs = plt.subplots(len(meshes), 4, figsize=(16, 4 * len(meshes)))
    for row, (piece, m) in zip(np.atleast_2d(axs), meshes):
        span = max(m.extents[:2]) + 4
        c = m.bounds.mean(0)
        for a, (t, R) in zip(row, views):
            items = [(m, -c, RESIN)] + bead_items(piece, -c)
            a.imshow(splat(items, R, span)); a.axis("off")
            a.set_title(f"{piece.title}: {t}", fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=80); plt.close()
    print("saved", out)


def render_hero(piece, m, out):
    """Large front and 3/4 views of the hero with its centre bead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = m.bounds.mean(0)
    span = max(m.extents[:2]) + 12
    fig, axs = plt.subplots(1, 3, figsize=(27, 9))
    for a, (t, R) in zip(axs, [("Front", np.eye(3)), ("3/4", rot("x", -0.5) @ rot("y", 0.45))]):
        a.imshow(splat([(m, -c, RESIN)] + bead_items(piece, -c), R, span, px=1000))
        a.axis("off"); a.set_title(f"Hero, {t}: {m.extents[0]:.0f} x {m.extents[1]:.0f} mm "
                                   f"(blue = Amy's 10 mm centre bead on the strand)")
    R = rot("x", -0.75) @ rot("y", 0.25)          # close-up, raking light shows the rails
    axs[2].imshow(splat([(m, np.array([0, 9.0, -4.0]), RESIN)], R, 26, px=1000))
    axs[2].axis("off"); axs[2].set_title(f"Close-up: rails round every petal and leaf, "
                                         f"{BED} mm epoxy beds")
    plt.tight_layout(); plt.savefig(out, dpi=80); plt.close()
    print("saved", out)


def render_strand(meshes, out):
    """The necklace front: secondary heroes either side of the hero, accents
    and placeholder 8 mm beads between."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    by = {p.name: (p, m) for p, m in meshes}
    order = ["bud", "leaf", "secondary", "hero", "secondary", "leaf", "bud"]
    flip = [False, False, False, False, True, True, True]
    gaps = [2, 3, 3, 3, 3, 2]                       # beads between consecutive pieces
    bead = trimesh.creation.icosphere(4, radius=4.0)
    items, x = [], 0.0
    for k, (nm, fl) in enumerate(zip(order, flip)):
        piece, m = by[nm]
        m = m.copy()
        if fl:
            m.apply_transform(np.diag([-1, 1, 1, 1])); m.fix_normals()
        yc, zc = piece.axis
        off = np.array([x - m.bounds[0][0], -yc, -zc])
        items.append((m, off, RESIN))
        items += bead_items(piece, off)
        x += m.extents[0]
        if k < len(order) - 1:
            for _ in range(gaps[k]):
                items.append((bead, np.array([x + 3.6, 0, 0]), BEAD_COL))
                x += 7.4
    for i in range(3):   # a few beads beyond each end
        items.append((bead, np.array([-3.6 - 7.4 * i, 0, 0]), BEAD_COL))
        items.append((bead, np.array([x + 3.6 + 7.4 * i, 0, 0]), BEAD_COL))
    span = x + 30
    img = splat(items, np.eye(3), span, px=1600, center=(x / 2, -8))
    h = img.shape[0]
    crop = img[int(h * 0.36):int(h * 0.66)]
    plt.figure(figsize=(20, 20 * crop.shape[0] / crop.shape[1]))
    plt.imshow(crop); plt.axis("off")
    plt.title("Necklace front laid flat: secondary heroes either side of the hero "
              "(blue = placeholder beads; Amy chooses the real ones)")
    plt.tight_layout(); plt.savefig(out, dpi=80); plt.close()
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.06)
    ap.add_argument("--only", nargs="*", help="piece names to build")
    ap.add_argument("--preview", action="store_true", help="write preview PNGs")
    ap.add_argument("--stl", action="store_true", help="write final STLs")
    args = ap.parse_args()

    series = [p for p in make_series() if not args.only or p.name in args.only]
    built, ok_all = [], True
    for p in series:
        t0 = time.time()
        m, n = mesh_sdf(p, args.voxel)
        m = decimate(m, args.voxel * 0.25)
        ok_all &= check(p, m, n)
        print(f"    built in {time.time() - t0:.0f}s")
        built.append((p, m))
        if args.stl:
            path = os.path.join(HERE, f"echo_{p.name}.stl")
            m.export(path); print("    wrote", path)
            if p.name == "secondary":                   # right-hand twin
                r = m.copy(); r.apply_transform(np.diag([-1, 1, 1, 1])); r.fix_normals()
                path = os.path.join(HERE, "echo_secondary_right.stl")
                r.export(path); print("    wrote", path)
    if args.preview:
        others = [b for b in built if b[0].name != "hero"]
        if others:
            render_pieces(others, os.path.join(HERE, "preview_pieces.png"))
        names = {p.name for p, _ in built}
        if "hero" in names:
            render_hero(*[b for b in built if b[0].name == "hero"][0],
                        os.path.join(HERE, "preview_hero.png"))
        if {"hero", "secondary", "leaf", "bud"} <= names:
            render_strand(built, os.path.join(HERE, "preview_strand.png"))
    if not ok_all:
        sys.exit("a piece failed its checks")


if __name__ == "__main__":
    main()
