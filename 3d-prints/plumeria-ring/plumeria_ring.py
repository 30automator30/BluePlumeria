#!/usr/bin/env python3
"""Hawaiian plumeria ring generator (resin / MSLA printing).

Builds the ring as a signed distance field (SDF), meshes it with marching
cubes, simplifies, and writes a watertight binary STL.

Design
  * Top:       large 5-petal plumeria, petals cupped with gently up-turned
               edges and tips, pinwheel overlap, tight swirled throat.
  * Band:      three vines twisted together like a rope, with small leaves
               budding off the vines on each shoulder, framed by a solid
               rail along each edge.
  * Inside:    a smooth comfort liner engraved with "ALOHA" at the back.

Axes (mm): the finger runs along Y, the flower faces +Z.

Usage
  python3 plumeria_ring.py --size 7
  python3 plumeria_ring.py --size 6.5 --voxel 0.05
  python3 plumeria_ring.py --size 7 --text ""   # no inner engraving
"""

import argparse
import time

import numpy as np

# ---------------------------------------------------------------- parameters

LINER_T = 0.6                   # inner comfort liner wall thickness
RAIL_HW, RAIL_GAP = 0.42, 0.12  # solid edge rails: half thickness, gap to vines
VINE_R = 0.62                   # radius of each of the three vines
VINE_ORBIT_R, VINE_ORBIT_Y = 0.55, 0.95  # twist ellipse (radial, along finger)
VINE_TWISTS = 9                 # full twists around the ring

MAIN_R = 7.2                    # main flower radius (~14.5 mm flower)
SIDE_R = 2.3                    # shoulder flower radius
SIDE_PHI = 1.05                 # shoulder flower angle from top (rad, ~60 deg)

TEXT_HEIGHT, TEXT_DEPTH = 1.2, 0.25


def us_size_to_inner_diameter(size):
    return 11.63 + 0.8128 * size


# ------------------------------------------------------------------ helpers

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def ellipsoid(px, py, pz, rx, ry, rz):
    q = np.sqrt((px / rx) ** 2 + (py / ry) ** 2 + (pz / rz) ** 2)
    return (q - 1.0) * min(rx, ry, rz)


_ts = np.linspace(0, 1, 2001)
_SHAPE_MAX = np.max(_ts ** 0.5 * np.sqrt(1 - _ts ** 4.0))


def petal_shape(t):
    """Obovate plumeria petal outline: narrow claw, broad rounded tip."""
    return t ** 0.5 * np.sqrt(np.maximum(0.0, 1 - t ** 4.0)) / _SHAPE_MAX


def flower(lx, ly, lz, radius, rot0, th_base, th_tip, n_petals=5):
    """Plumeria SDF in a local frame: flower base at lz=0, facing +lz.

    Designed at a 7.5 mm reference radius and uniformly scaled; petal
    thickness is given in absolute mm so small flowers stay printable.
    """
    s = radius / 7.5
    x, y, z = lx / s, ly / s, lz / s
    tb, tt = th_base / s, th_tip / s
    L, wmax, off = 7.5, 3.2, 1.0

    d = np.full(x.shape, 1e9, dtype=np.float32)
    for i in range(n_petals):
        th = rot0 + 2 * np.pi * i / n_petals
        dx, dy = np.cos(th), np.sin(th)
        nx, ny = -dy, dx
        px, py = x - off * nx, y - off * ny        # offset axis -> pinwheel
        u = px * dx + py * dy
        v = px * nx + py * ny
        t = np.clip(u / L, 0.0, 1.0)
        w = wmax * petal_shape(t) * np.where(v > 0, 1.1, 0.9) + 1e-3
        d2 = np.maximum(np.abs(v) - w, np.maximum(-u, u - L))

        # cupped petal: rises from the throat, edges and tip curl gently up
        tilt = 0.18 * t                            # twist -> overlapping edges
        cup = 0.6 * t * (v / wmax) ** 2
        rim = 0.35 * t * np.exp(-np.maximum(w - np.abs(v), 0.0) / 0.5)
        tip = np.maximum(0.0, t - 0.55)
        h = (1.5 * (1 - np.exp(-4.0 * t)) + 0.15 * t + 1.6 * tip ** 2
             + tilt * v + cup + rim)
        dh_du = (6.0 * np.exp(-4.0 * t) + 0.15 + 3.2 * tip) / L
        dh_dv = tilt + 1.2 * t * v / wmax ** 2
        g = np.minimum(np.sqrt(1 + dh_du ** 2 + dh_dv ** 2), 1.6)
        thick = tt + (tb - tt) * (1 - t)
        ds = np.abs(z - h) / g - thick / 2
        d = smin(d, np.maximum(d2, ds), 0.12)

    # throat: solid core with a recessed funnel in the middle
    rr = np.sqrt(x * x + y * y)
    core = np.maximum(rr - (2.2 - 0.35 * z), np.maximum(-z - 0.8, z - 1.6))
    d = smin(d, core, 0.4)
    funnel = np.maximum(rr - (0.3 + 0.5 * (z - 0.6)), 0.6 - z)
    d = np.maximum(d, -funnel)
    return d * s


def leaf(s, y, n, s0, y0, ang, length, half_w, height):
    """Small raised leaf in band-surface coords (s=arc, n=height)."""
    ca, sa = np.cos(ang), np.sin(ang)
    u = (s - s0) * ca + (y - y0) * sa
    v = -(s - s0) * sa + (y - y0) * ca
    t = np.clip(u / length, 0.0, 1.0)
    w = half_w * np.sin(np.pi * t ** 0.8) ** 0.85 + 1e-3
    d2 = np.maximum(np.abs(v) - w, np.maximum(-u, u - length))
    vn = np.clip(np.abs(v) / w, 0.0, 1.0)
    hl = height * (0.45 + 0.55 * np.sqrt(1 - vn ** 2)) + 0.25 * t   # tip lifts
    hl = hl - 0.09 * np.exp(-(v / 0.08) ** 2)                        # midrib
    return np.maximum(np.maximum(d2, np.abs(n - hl / 2) - hl / 2 - 0.25), -n - 0.6)


def text_sdf_2d(text, height, res=0.01):
    """2D signed distance (mm) of `text`, returned as a sampler function."""
    if not text:
        return None
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath
    from scipy.ndimage import distance_transform_edt

    fp = FontProperties(family="DejaVu Sans", weight="bold")
    tp = TextPath((0, 0), text, size=1.0, prop=fp)
    verts = tp.vertices
    cap = TextPath((0, 0), "H", size=1.0, prop=fp).vertices[:, 1].max()
    scale = height / cap
    xmin, ymin = verts.min(0)
    xmax, ymax = verts.max(0)
    cx, cy = (xmin + xmax) / 2, cap / 2

    pad = 0.3
    xs = np.arange((xmin - cx) * scale - pad, (xmax - cx) * scale + pad, res)
    ys = np.arange((ymin - cy) * scale - pad, (ymax - cy) * scale + pad, res)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([gx.ravel() / scale + cx, gy.ravel() / scale + cy])
    inside = np.zeros(len(pts), bool)
    for poly in tp.to_polygons():          # even-odd fill handles counters
        inside ^= Path(poly).contains_points(pts)
    inside = inside.reshape(gx.shape)
    sd = (distance_transform_edt(~inside) - distance_transform_edt(inside)) * res

    def sample(px, py):
        fx = (px - xs[0]) / res
        fy = (py - ys[0]) / res
        ok = (fx >= 0) & (fx < len(xs) - 1) & (fy >= 0) & (fy < len(ys) - 1)
        ix = np.clip(fx.astype(int), 0, len(xs) - 2)
        iy = np.clip(fy.astype(int), 0, len(ys) - 2)
        ax, ay = fx - ix, fy - iy
        val = (sd[ix, iy] * (1 - ax) * (1 - ay) + sd[ix + 1, iy] * ax * (1 - ay)
               + sd[ix, iy + 1] * (1 - ax) * ay + sd[ix + 1, iy + 1] * ax * ay)
        return np.where(ok, val, 1.0)

    return sample


# --------------------------------------------------------------------- ring

class Ring:
    def __init__(self, size, text="ALOHA"):
        self.ri = us_size_to_inner_diameter(size) / 2
        self.rc = self.ri + LINER_T + VINE_ORBIT_R + VINE_R - 0.45  # vine axis
        self.rout = self.rc + VINE_ORBIT_R + VINE_R                  # vine crest
        self.text = text_sdf_2d(text, TEXT_HEIGHT)

    def sdf(self, x, y, z):
        ri, rc = self.ri, self.rc
        r = np.sqrt(x * x + z * z)
        phi = np.arctan2(x, z)                  # 0 at top, +-pi at back

        # vines (and the rails framing them) fan out a little under the flower
        spread = 1.0 + 0.35 * np.exp(-(phi / 0.7) ** 2)
        rail_y = VINE_ORBIT_Y * spread + VINE_R + RAIL_GAP + RAIL_HW

        # smooth comfort liner (holds the engraving, keeps the inside comfy)
        qr = np.abs(r - (ri + LINER_T / 2)) - (LINER_T / 2 - 0.25)
        qy = np.abs(y) - (rail_y - 0.25)
        d = (np.sqrt(np.maximum(qr, 0) ** 2 + np.maximum(qy, 0) ** 2)
             + np.minimum(np.maximum(qr, qy), 0) - 0.25)

        # solid rounded rails along both edges, just below the vine crests
        rail_t = self.rout - ri - 0.2
        qr = np.abs(r - (ri + rail_t / 2)) - (rail_t / 2 - 0.35)
        qy = np.abs(np.abs(y) - rail_y) - (RAIL_HW - 0.35)
        rail = (np.sqrt(np.maximum(qr, 0) ** 2 + np.maximum(qy, 0) ** 2)
                + np.minimum(np.maximum(qr, qy), 0) - 0.35)
        d = smin(d, rail, 0.1)

        # three twisted vines between the rails
        for k in range(3):
            psi = VINE_TWISTS * phi + 2 * np.pi * k / 3
            cr = rc + VINE_ORBIT_R * np.cos(psi)
            cy = VINE_ORBIT_Y * spread * np.sin(psi)
            vine = np.sqrt((r - cr) ** 2 + (y - cy) ** 2) - VINE_R
            d = smin(d, vine, 0.18)

        # little leaves budding off the vines along both shoulders
        s = np.abs(phi) * self.rout
        n = r - self.rout + 0.25
        for (s0, y0, ang, ln, hw) in [
            (5.2, 0.4, 0.40, 2.8, 0.70),
            (6.2, -0.4, -0.42, 2.6, 0.66),
            (12.6, 0.35, 0.36, 2.5, 0.60),
            (13.6, -0.35, -0.38, 2.4, 0.58),
            (16.5, 0.3, 0.34, 2.2, 0.55),
            (17.3, -0.3, -0.34, 2.0, 0.52),
        ]:
            d = smin(d, leaf(s, y, n, s0, y0, ang, ln, hw, 0.45), 0.15)

        # main flower on a sculpted calyx
        zb = self.rout - 0.1
        calyx = ellipsoid(x, y, z - (zb - 0.7), 4.6, 3.6, 2.0)
        d = smin(d, calyx, 0.9)
        d = smin(d, flower(x, y, z - zb, MAIN_R, 0.31, 1.15, 0.8), 0.45)

        # shoulder flowers, each rotated so it faces radially outward
        for sgn, rot in [(1, 1.1), (-1, -0.4)]:
            a = sgn * SIDE_PHI
            lx = x * np.cos(a) - z * np.sin(a)
            lz = x * np.sin(a) + z * np.cos(a)
            base = self.rout - 0.25
            cal = ellipsoid(lx, y, lz - (base - 0.2), 2.1, 2.1, 1.0)
            d = smin(d, cal, 0.4)
            d = smin(d, flower(lx, y, lz - base, SIDE_R, rot, 0.75, 0.6), 0.25)

        # engraved text inside the back of the liner
        if self.text is not None:
            phiw = np.where(phi < 0, phi + 2 * np.pi, phi)
            si = -(phiw - np.pi) * ri
            dt = np.maximum(self.text(si, y), r - (ri + TEXT_DEPTH))
            d = np.maximum(d, -dt)

        # keep the finger hole perfectly clean
        return np.maximum(d, ri - r)


# ------------------------------------------------------------------ meshing

def build_mesh(ring, voxel, y_half=MAIN_R + 0.8, z_top=None, pad=2.5):
    from skimage.measure import marching_cubes
    import trimesh

    ext = ring.rout + pad
    z_top = ring.rout + 3.8 if z_top is None else z_top
    xs = np.arange(-ext, ext + voxel, voxel, dtype=np.float32)
    ys = np.arange(-y_half, y_half + voxel, voxel, dtype=np.float32)
    zs = np.arange(-(ring.rout + 0.6), z_top + voxel, voxel, dtype=np.float32)
    print(f"grid {len(xs)} x {len(ys)} x {len(zs)} = {len(xs)*len(ys)*len(zs)/1e6:.1f}M voxels")

    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    gy, gz = np.meshgrid(ys, zs, indexing="ij")
    t0 = time.time()
    for i, xv in enumerate(xs):
        vol[i] = ring.sdf(np.full_like(gy, xv), gy, gz)
        if i % 100 == 0:
            print(f"  slice {i}/{len(xs)}  {time.time()-t0:.0f}s", flush=True)
    vol = np.pad(vol, 1, constant_values=1.0)

    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(voxel,) * 3)
    verts -= voxel
    verts += np.array([xs[0], ys[0], zs[0]])
    mesh = trimesh.Trimesh(verts, faces, process=True)
    # discard any stray specks so the STL is a single solid
    parts = mesh.split(only_watertight=False)
    return max(parts, key=lambda m: len(m.faces))


def decimate(mesh, tolerance):
    """Simplify within `tolerance` mm while keeping the mesh a closed solid.

    Manifold's simplifier keeps topology intact, but can leave two distinct
    vertices at the same spot where thin features touch; STL stores no
    topology, so those would fuse on reload. Nudge such twins apart by 1 um
    along their own normals so the saved STL stays watertight.
    """
    import manifold3d
    import trimesh

    solid = manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, np.float32),
        tri_verts=np.asarray(mesh.faces, np.uint32)))
    out = solid.simplify(tolerance).to_mesh()
    m = trimesh.Trimesh(out.vert_properties[:, :3].astype(np.float64),
                        out.tri_verts, process=False)
    _, idx, counts = np.unique(np.round(m.vertices, 6), axis=0,
                               return_inverse=True, return_counts=True)
    twins = counts[idx.ravel()] > 1
    if twins.any():
        v = m.vertices.copy()
        v[twins] -= 0.001 * m.vertex_normals[twins]
        m = trimesh.Trimesh(v, m.faces, process=False)
    # simplification can leave a zero-volume speck of a few triangles
    return max(m.split(only_watertight=False), key=lambda p: len(p.faces))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=7.0, help="US ring size")
    ap.add_argument("--voxel", type=float, default=0.06, help="mesh resolution, mm")
    ap.add_argument("--tolerance", type=float, default=0.004,
                    help="max simplification deviation, mm")
    ap.add_argument("--text", default="ALOHA", help='inner engraving ("" for none)')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ring = Ring(args.size, args.text)
    print(f"US size {args.size}: inner diameter {2*ring.ri:.2f} mm")
    mesh = build_mesh(ring, args.voxel)
    print(f"marching cubes: {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
    mesh = decimate(mesh, args.tolerance)
    print(f"final: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume/1000:.3f} mL, bounds={np.round(mesh.extents, 2)} mm")
    out = args.out or f"plumeria_ring_US{args.size:g}.stl"
    mesh.export(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
