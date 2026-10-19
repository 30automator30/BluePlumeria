#!/usr/bin/env python3
"""Self-supporting gyroid plumeria pendant (resin / MSLA, Saturn 4 Ultra).

Designed so it needs no supports: it prints standing upright, exactly as
it hangs. Each of the five petals is filled with gyroid lace, a triply
periodic minimal surface whose walls always rise smoothly with no flat
ceilings, so every layer grows out of the one below. The lace is graded
finer toward the centre, and every petal carries an identical copy of it
(five-fold symmetry). Rounded rims frame the outline and the seams between
the pinwheel petals, a domed bead sits at the centre, and the bail is a
teardrop (pointed arch) so its top never overhangs flat. The only extra
geometry is a breakaway foot under the lowest petal tips, which also takes
the squashed first layers.

After meshing, the part is re-sliced and checked for islands (regions
that start in mid-air) and steep overhangs.

Axes (mm): flower in the XZ plane, Y through the thickness, Z up (as
printed and as worn).

Usage
  python3 gyroid_pendant.py                 # 40 mm pendant
  python3 gyroid_pendant.py --size 32       # smaller
  python3 gyroid_pendant.py --voxel 0.08    # quick look
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plumeria-ring"))
from plumeria_ring import decimate, petal_shape  # noqa: E402

RES2 = 0.02          # 2D raster resolution for the outline, mm


class Pendant:
    def __init__(self, size=40.0, thick=2.6, rot=None, wall=0.55, style="pointed",
                 cell_scale=1.0):
        self.R = size / 2 * 0.86             # petal length (flower ~size wide)
        self.T = thick
        self.style = style
        self.wall = wall
        self.cell_scale = cell_scale         # >1: larger gyroid cells, wider pores
        if rot is None:
            rot = self.best_rotation()
        self.rot = np.radians(rot)           # direction of the top petal
        self.rot_deg = rot
        self._outline()

    # ------------------------------------------------------- petal outline
    def petal_masks(self, rot, X, Z):
        """Label map of the five pinwheel petals (0 = outside)."""
        R = self.R
        label = np.zeros(X.shape, np.int16)
        for i in range(5):
            th = rot + 2 * np.pi * i / 5
            dx, dz = np.cos(th), np.sin(th)
            if self.style == "round":
                off = 0.10 * R
                px, pz = X + off * dz, Z - off * dx
                u, v = px * dx + pz * dz, -px * dz + pz * dx
                t = np.clip(u / R, 0, 1)
                w = 0.36 * R * petal_shape(t) * np.where(v > 0, 1.1, 0.9)
            else:
                # pointed plumeria petal: narrow claw, broad belly, sharp
                # tip; the centreline sweeps sideways (pinwheel twist)
                off = 0.14 * R
                px, pz = X + off * dz, Z - off * dx
                u, v = px * dx + pz * dz, -px * dz + pz * dx
                t = np.clip(u / R, 0, 1)
                v = v - 0.16 * R * t ** 2                     # sweep
                w = 0.29 * R * t ** 0.6 * (1 - t) ** 0.8 / 0.36
                w = w * np.where(v > 0, 1.12, 0.88)
            m = (np.abs(v) < w) & (u > 0) & (u < R)
            label[m] = i + 1                                  # each over the last
        return label

    def best_rotation(self, res=0.08):
        """Choose the rotation that leaves the fewest low points, puts a
        petal tip straight up for the bail and keeps the bottom tips level."""
        ext = self.R + 1.5
        g = np.arange(-ext, ext, res)
        X, Z = np.meshgrid(g, g, indexing="ij")
        best = None
        for deg in np.arange(0, 72, 1.0):
            inside = self.petal_masks(np.radians(deg), X, Z) > 0
            inside |= np.hypot(X, Z) < 0.2 * self.R
            lows = []
            for k in range(1, inside.shape[1]):
                lab, n = ndimage.label(inside[:, k])
                if n:
                    hit = ndimage.maximum(inside[:, k - 1], lab, np.arange(1, n + 1))
                    lows += [g[k]] * int((hit == 0).sum())
            cols = np.nonzero(inside.any(axis=1))[0]
            topk = max(np.nonzero(inside[c])[0].max() for c in cols)
            top_x = g[np.nonzero(inside[:, topk])[0]].mean()
            spread = (max(lows) - min(lows)) if lows else 0
            score = 10 * len(lows) + abs(top_x) + 0.5 * spread
            if best is None or score < best[0]:
                best = (score, deg, len(lows), top_x, spread)
        print(f"rotation {best[1]:.0f} deg: {best[2]} low points, top tip at x={best[3]:+.2f} mm, "
              f"bottom tips within {best[4]:.2f} mm")
        return float(best[1])

    # ------------------------------------------------------------ 2D maps
    def _outline(self):
        R = self.R
        ext = R + 1.5
        self.g = np.arange(-ext, ext, RES2)
        X, Z = np.meshgrid(self.g, self.g, indexing="ij")
        label = self.petal_masks(self.rot, X, Z)
        hub = np.hypot(X, Z) < 0.2 * R
        solid = (label > 0) | hub
        self.top_z = float(Z[solid].max()) - 0.4
        self.top_x = float(X[solid & (Z >= Z[solid].max() - 0.05)].mean())
        label[hub & (label == 0)] = 1
        inside = label > 0
        self.sd = (ndimage.distance_transform_edt(~inside)
                   - ndimage.distance_transform_edt(inside)) * RES2
        # seams between petals
        edge = np.zeros(X.shape, bool)
        edge[1:] |= (label[1:] != label[:-1]) & (label[1:] > 0) & (label[:-1] > 0)
        edge[:, 1:] |= (label[:, 1:] != label[:, :-1]) & (label[:, 1:] > 0) & (label[:, :-1] > 0)
        self.seam = ndimage.distance_transform_edt(~edge) * RES2
        self.label = label
        # every low point of the outline (where printing would start in
        # mid-air): each gets a neck down to the breakaway foot
        self.low_points = []
        for k in range(1, inside.shape[1]):
            lab, n = ndimage.label(inside[:, k])
            if n:
                hit = ndimage.maximum(inside[:, k - 1], lab, np.arange(1, n + 1))
                for c in np.nonzero(hit == 0)[0]:
                    xs_ = np.nonzero(lab == c + 1)[0]
                    self.low_points.append((float(self.g[int(xs_.mean())]), float(self.g[k])))

    def _sample(self, arr, x, z, order=1):
        ix = (x - self.g[0]) / RES2
        iz = (z - self.g[0]) / RES2
        return ndimage.map_coordinates(arr, [ix, iz], order=order, mode="nearest")

    # ---------------------------------------------------------------- SDF
    def sdf(self, x, y, z):
        R, T = self.R, self.T
        sd = self._sample(self.sd, x, z)
        seam = self._sample(self.seam, x, z)
        lab = self._sample(self.label.astype(np.float32), x, z, order=0).astype(int)
        r = np.hypot(x, z)

        # gyroid in each petal's own frame (identical lace in every petal),
        # cell period graded from 2.3 mm at the centre to 3.6 mm at the tips
        ang = -(self.rot + 2 * np.pi * (np.maximum(lab, 1) - 1) / 5)
        u = x * np.cos(ang) - z * np.sin(ang)
        v = x * np.sin(ang) + z * np.cos(ang)
        P = (2.3 + 1.3 * np.clip(r / R, 0, 1)) * self.cell_scale
        k = 2 * np.pi / P
        # integrate the varying frequency radially so cells stay unbroken
        phase_u = 2 * np.pi * (u / P)
        g = (np.sin(phase_u) * np.cos(k * v) + np.sin(k * v) * np.cos(k * y)
             + np.sin(k * y) * np.cos(phase_u))
        t = np.clip(self.wall / (0.265 * P), 0.3, 1.2)        # wall ~ self.wall mm
        lace = (np.abs(g) - t) / (1.2 * k)
        slab = np.abs(y) - T / 2
        body = np.maximum(np.maximum(lace, sd), slab)

        # rounded rim along the outline and the petal seams
        rim_w, seam_w = 0.95, 0.65
        q = np.maximum(np.abs(sd + rim_w / 2) - rim_w / 2, 0)
        rim = np.sqrt(q ** 2 + np.maximum(np.abs(y) - (T / 2 - 0.45), 0) ** 2) - 0.45
        rim = np.maximum(rim, sd)
        sq = np.maximum(seam - seam_w / 2, 0)
        seams = np.sqrt(sq ** 2 + np.maximum(np.abs(y) - (T / 2 - 0.55), 0) ** 2) - 0.32
        seams = np.maximum(seams, sd)
        d = np.minimum(np.minimum(body, rim), seams)

        # domed bead at the centre, with a raised ring
        bead_r = 0.16 * R
        bead = np.sqrt(r ** 2 + (np.abs(y) * bead_r / (T / 2 + 0.5)) ** 2) - bead_r
        d = np.minimum(d, bead)

        # teardrop bail above the top petal tip, in the YZ plane
        top = self.top_z
        bz = z - (top + 2.6)
        hole = teardrop(y, bz, 1.45)
        outer = teardrop(y, bz, 1.45 + 0.9)
        bx = x - self.top_x
        bail = np.maximum(np.maximum(outer, -hole), np.abs(bx) - 0.8)
        neck = np.maximum(np.hypot(bx, y) - 0.9, np.maximum(-(z - top + 1.5), z - (top + 1.2)))
        d = np.minimum(d, np.minimum(bail, neck))
        return d


def teardrop(y, z, r):
    """2D teardrop: a circle of radius r with a pointed roof whose sides
    rise at 45 degrees to an apex r*sqrt(2) above the centre (signed
    distance, exact enough for meshing)."""
    circ = np.hypot(y, z) - r
    apex = r * np.sqrt(2)
    roof = (np.abs(y) + z - apex) / np.sqrt(2)
    tri = np.maximum(roof, np.maximum(-z, np.abs(y) - r))
    return np.minimum(circ, tri)


# ------------------------------------------------------------- foot + mesh

def add_foot(vol, xs, ys, zs, pendant):
    """Breakaway foot: a pad below the flower, with a thin neck rising to
    every low point of the outline, so only the foot touches the plate."""
    lows = pendant.low_points
    zbot = min(z for _, z in lows)
    pad_top = zbot - 1.8
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    d = np.full(vol.shape, 1e3, np.float32)
    for tx, tz in lows:
        h = np.clip((Z - pad_top) / (tz - pad_top), 0, 1)
        rr = 1.0 - 0.62 * h                                  # 0.38 mm at the tip
        neck = np.maximum(np.hypot(X - tx, Y) - rr,
                          np.maximum(pad_top - Z, Z - (tz + 0.35)))
        d = np.minimum(d, neck)
    txs = [x for x, _ in lows]
    lo_x, hi_x = min(txs) - 3.0, max(txs) + 3.0
    pad = np.maximum(np.maximum(lo_x - X, X - hi_x),
                     np.maximum(np.abs(Y) - 2.5, np.maximum(Z - pad_top, (pad_top - 1.0) - Z)))
    d = np.minimum(d, pad)
    return np.minimum(vol, d), len(lows)


def open_solid(vol, voxel, r):
    """Morphological opening (erode then dilate by r) of the solid in an
    SDF volume. Where the flat front/back faces slice through a gyroid wall
    that only grazes them, they leave hair-thin slivers whose lower tips
    would print as islands; opening removes everything thinner than 2r and
    rounds the cut edges, while walls, rims and bead are kept."""
    eroded = vol < -r
    out = (ndimage.distance_transform_edt(~eroded)
           - ndimage.distance_transform_edt(eroded)) * voxel
    return (out - r).astype(np.float32)


def fillet_for(island, solid, xs, ys, zs):
    """A 45-degree strut from a face island down and inward to the nearest
    solid within the cone below it (design coordinates)."""
    x0, y0, z0 = island
    vz = zs[1] - zs[0]
    best = None
    for dz in np.arange(0.15, 2.5, vz):
        k = int(round((z0 - dz - zs[0]) / vz))
        if k < 0:
            break
        reach = dz                                          # 45-degree cone
        i0, i1 = np.searchsorted(xs, x0 - reach), np.searchsorted(xs, x0 + reach)
        j0, j1 = np.searchsorted(ys, y0 - reach), np.searchsorted(ys, y0 + reach)
        sub = solid[i0:i1, j0:j1, k]
        if sub.any():
            ii, jj = np.nonzero(sub)
            px, py = xs[i0 + ii], ys[j0 + jj]
            dist = np.hypot(px - x0, py - y0)
            ok = dist <= reach
            if ok.any():
                m = np.argmin(np.where(ok, dist, 1e9))
                best = (float(px[m]), float(py[m]), float(zs[k]) + 0.1)
                break
    return best


def capsule(X, Y, Z, a, b, r):
    a, b = np.asarray(a), np.asarray(b)
    ab = b - a
    t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1] + (Z - a[2]) * ab[2]) / (ab @ ab), 0, 1)
    return np.sqrt((X - a[0] - t * ab[0]) ** 2 + (Y - a[1] - t * ab[1]) ** 2
                   + (Z - a[2] - t * ab[2]) ** 2) - r


def build(p, voxel, fillets=(), want_grid=False):
    from skimage.measure import marching_cubes
    import trimesh
    ext = p.R + 1.5
    xs = np.arange(-ext, ext, voxel, dtype=np.float32)
    ys = np.arange(-p.T / 2 - 3.0, p.T / 2 + 3.0, voxel, dtype=np.float32)
    zs = np.arange(-ext - 4.0, p.R + 7.5, voxel, dtype=np.float32)
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    gy, gz = np.meshgrid(ys, zs, indexing="ij")
    for i, xv in enumerate(xs):
        vol[i] = np.minimum(p.sdf(np.full_like(gy, xv), gy, gz), 3.0)
    vol = open_solid(vol, voxel, 0.15)          # no slivers thinner than 0.3 mm
    for a, b in fillets:                         # micro-fillets under face islands
        lo = np.minimum(a, b) - 0.5
        hi = np.maximum(a, b) + 0.5
        i0, i1 = np.searchsorted(xs, lo[0]), np.searchsorted(xs, hi[0])
        j0, j1 = np.searchsorted(ys, lo[1]), np.searchsorted(ys, hi[1])
        k0, k1 = np.searchsorted(zs, lo[2]), np.searchsorted(zs, hi[2])
        X, Y, Z = np.meshgrid(xs[i0:i1], ys[j0:j1], zs[k0:k1], indexing="ij")
        vol[i0:i1, j0:j1, k0:k1] = np.minimum(vol[i0:i1, j0:j1, k0:k1],
                                              capsule(X, Y, Z, a, b, 0.18))
    grid = (vol < 0, xs, ys, zs) if want_grid else None
    vol, n_tips = add_foot(vol, xs, ys, zs, p)
    vol = np.pad(vol, 1, constant_values=1.0)
    vol[vol == 0] = 1e-6
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(voxel,) * 3)
    verts += np.array([xs[0], ys[0], zs[0]]) - voxel
    mesh = trimesh.Trimesh(verts, faces, process=False)
    parts = mesh.split(only_watertight=False)
    big = max(parts, key=lambda m: len(m.faces))
    dropped = sum(abs(q.volume) for q in parts if q is not big)
    mesh = decimate(big, 0.004)
    shift = -mesh.bounds[0, 2]
    mesh.apply_translation([0, 0, shift])                         # foot on the plate
    return mesh, n_tips, len(parts) - 1, dropped, shift, grid


# ------------------------------------------------------------ verification

def verify(mesh, layer=0.05, xy=0.025):
    """Slice like a printer and look for trouble."""
    import manifold3d as mf
    from skimage.draw import polygon as fill_polygon
    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                            tri_verts=np.asarray(mesh.faces, np.uint32)))
    b = mesh.bounds
    lo = b[0] - [0.5, 0.5, 0]
    shape = tuple(np.ceil((b[1] + [0.5, 0.5, 0] - lo)[:2] / xy).astype(int))
    nz = int(np.ceil((b[1, 2] - b[0, 2]) / layer))
    prev = None
    islands, cantilever = [], 0
    reach = max(1, int(round(layer / np.tan(np.radians(30)) / xy)))  # 30 deg
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(nz):
        z = (k + 0.5) * layer
        cur = np.zeros(shape, bool)
        for poly in m.slice(z).to_polygons():
            p = np.asarray(poly)
            if len(p) >= 3:
                rr, cc = fill_polygon((p[:, 0] - lo[0]) / xy, (p[:, 1] - lo[1]) / xy, shape)
                cur[rr, cc] ^= True
        if prev is not None and k > 2:
            lab, n = ndimage.label(cur)
            if n:
                idx = np.arange(1, n + 1)
                # detached = nothing within 2 px (50 um) below; single raster
                # pixels are slicing noise, not geometry
                near = ndimage.binary_dilation(prev, st, iterations=2)
                hit = ndimage.maximum(near, lab, idx)
                area = ndimage.sum(cur, lab, idx) * xy * xy
                cen = ndimage.center_of_mass(cur, lab, idx)
                for c in np.nonzero((hit == 0) & (area >= 0.01))[0]:
                    islands.append((round(z, 2), round(float(area[c]), 3),
                                    round(lo[0] + cen[c][0] * xy, 2), round(lo[1] + cen[c][1] * xy, 2)))
            grown = ndimage.binary_dilation(prev, st, iterations=reach)
            cantilever = max(cantilever, int((cur & ~grown).sum()))
        prev = cur
    # downward-facing surface flatter than 30 degrees from horizontal
    n = mesh.face_normals
    flat_down = mesh.area_faces[(n[:, 2] < -np.cos(np.radians(30))) &
                                (mesh.triangles_center[:, 2] > 2.0)].sum()
    return islands, cantilever * xy * xy, flat_down, mesh.area


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=40.0, help="flower width, mm")
    ap.add_argument("--rot", type=float, default=None,
                    help="direction of the top petal, deg (default: searched for the "
                         "fewest low points; v1 used 84 with --style round)")
    ap.add_argument("--style", choices=["pointed", "round"], default="pointed")
    ap.add_argument("--thick", type=float, default=2.6, help="pendant thickness, mm (v1: 3.6)")
    ap.add_argument("--voxel", type=float, default=0.04)
    ap.add_argument("--metal", action="store_true",
                    help="for casting or electroplating: 0.8 mm walls, 1.4x larger cells "
                         "so the pores stay at least 1 mm open")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    p = Pendant(args.size, thick=args.thick, rot=args.rot, style=args.style,
                wall=0.8 if args.metal else 0.55, cell_scale=1.4 if args.metal else 1.0)
    fillets = []
    for attempt in range(1, 5):
        mesh, n_tips, n_frag, dropped, shift, grid = build(p, args.voxel, fillets, True)
        print(f"pass {attempt}: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
              f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, "
              f"foot under {n_tips} tip(s), {len(fillets)} micro-fillets, "
              f"dropped {n_frag} fragments ({dropped:.3f} mm3), {time.time() - t0:.0f}s")
        islands, cant, flat, area = verify(mesh)
        print(f"  verify: {len(islands)} islands")
        if not islands:
            break
        solid, xs, ys, zs = grid
        added = 0
        for z, a_, x, y in islands:
            top = (x, y, z - shift)
            foot = fillet_for(top, solid, xs, ys, zs)
            if foot:
                fillets.append((np.array([x, y, z - shift + 0.08]), np.array(foot)))
                added += 1
        if not added:
            break
    print(f"verify: {len(islands)} islands {islands[:10]}; worst per-layer overhang "
          f"beyond 30 deg: {cant:.3f} mm2; downward faces flatter than 30 deg: "
          f"{flat:.1f} of {area:.0f} mm2 ({100 * flat / area:.2f}%)")
    # a lighter file for sharing, kept only if it still verifies clean
    # (simplifying can shave a micro-fillet and bring an island back)
    if not islands:
        for tol in (0.008, 0.006):
            light = decimate(mesh, tol)
            li, *_ = verify(light)
            print(f"  simplified at {tol} mm: {len(light.faces)} faces, "
                  f"watertight={light.is_watertight}, {len(li)} islands")
            if light.is_watertight and not li:
                mesh = light
                break
    out = args.out or f"gyroid_plumeria_pendant_{args.size:g}mm{'_metal' if args.metal else ''}.stl"
    mesh.export(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
