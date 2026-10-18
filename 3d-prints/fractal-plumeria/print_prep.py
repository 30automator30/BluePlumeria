#!/usr/bin/env python3
"""Orient and add supports to the lattice plumeria parts (Saturn 4 Ultra).

For each part:
  1. orient it for printing (body upright on its root curls; bloom face up,
     tilted, on a raft) and place it on the build plate;
  2. slice it into layers and find every island (a region that starts in
     mid-air) and every overhang steeper than ~45 deg;
  3. put a support under each, spaced on a grid: a fine contact tip, a
     shaft that thickens with height, a foot on the plate or raft (or a
     second tip where the shaft lands on the model), with cross braces
     between tall neighbouring shafts;
  4. union model + supports into one watertight STL, re-slice it and
     confirm that no layer starts in mid-air.

Usage
  python3 print_prep.py --body BODY.stl --bloom BLOOM.stl --outdir OUT
"""

import argparse
import os
import time

import numpy as np
import manifold3d as mf
import trimesh
from scipy import ndimage
from skimage.draw import polygon as fill_polygon

PLATE_X, PLATE_Y, PLATE_Z = 218.88, 122.88, 220.0   # Saturn 4 Ultra volume
XY = 0.2            # planning grid, mm
DZ = 0.2            # planning layer height, mm
SPACING = 3.0       # support grid spacing under overhangs, mm
Z_BIN = 2.0         # vertical bin: one contact per 3 x 3 x 2 mm cell
LIFT = 6.0          # bloom lift above the raft, mm
RAFT_T = 1.2


# ------------------------------------------------------------- geometry io

def to_manifold(mesh):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                               tri_verts=np.asarray(mesh.faces, np.uint32)))


def to_trimesh(m):
    me = m.to_mesh()
    return trimesh.Trimesh(me.vert_properties[:, :3], me.tri_verts, process=False)


def rotation(axis, deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), axis)


# ----------------------------------------------------------------- slicing

def voxelize(m, lo, shape):
    """Slice manifold m into a boolean [x, y, z] grid (cell XY, layer DZ),
    sampling each layer at its mid-height, even-odd filled."""
    occ = np.zeros(shape, bool)
    for k in range(shape[2]):
        z = lo[2] + (k + 0.5) * DZ
        cs = m.slice(z)
        layer = np.zeros(shape[:2], bool)
        for poly in cs.to_polygons():
            p = np.asarray(poly)
            if len(p) < 3:
                continue
            rr, cc = fill_polygon((p[:, 0] - lo[0]) / XY, (p[:, 1] - lo[1]) / XY, shape[:2])
            layer[rr, cc] ^= True
        occ[:, :, k] = layer
    return occ


def grid_for(m, pad=3.0):
    b = np.array(m.bounding_box()).reshape(2, 3)
    lo = np.array([b[0, 0] - pad, b[0, 1] - pad, 0.0])
    hi = np.array([b[1, 0] + pad, b[1, 1] + pad, b[1, 2] + DZ])
    shape = tuple(np.ceil((hi - lo) / [XY, XY, DZ]).astype(int))
    return lo, shape


def unsupported(occ, reach=1):
    """Cells that start in mid-air: solid, with nothing within `reach`
    cells (a ~45 deg cone) in the layer below."""
    out = np.zeros_like(occ)
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(1, occ.shape[2]):
        below = ndimage.binary_dilation(occ[:, :, k - 1], st, iterations=reach)
        out[:, :, k] = occ[:, :, k] & ~below
    return out


# ---------------------------------------------------------------- supports

def plan_contacts(occ, lo, min_z):
    """Support contact points under every island and overhang."""
    need = unsupported(occ)
    need[:, :, :int(min_z / DZ) + 1] = False          # resting on the plate
    lab, n = ndimage.label(need, structure=np.ones((3, 3, 3)))
    pts = []
    cell = max(1, int(SPACING / XY))
    if n:
        idx = np.argwhere(need)
        zc = max(1, int(Z_BIN / DZ))
        key = np.column_stack([idx[:, 0] // cell, idx[:, 1] // cell, idx[:, 2] // zc])
        # lowest unsupported cell in each 3D bin (so higher overhangs above
        # an already-supported one still get their own support) ...
        order = np.lexsort((idx[:, 2], key[:, 2], key[:, 1], key[:, 0]))
        idx, key = idx[order], key[order]
        first = np.ones(len(idx), bool)
        first[1:] = np.any(key[1:] != key[:-1], axis=1)
        chosen = {tuple(i) for i in idx[first]}
        # ... plus the lowest cell of every island component
        lows = ndimage.minimum_position(np.where(need, np.arange(need.shape[2])[None, None, :], 1e9),
                                        lab, index=np.arange(1, n + 1))
        chosen |= {tuple(int(v) for v in p) for p in lows}
        for i, j, k in chosen:
            pts.append(lo + np.array([(i + 0.5) * XY, (j + 0.5) * XY, k * DZ]))
    return np.array(pts), n


def shaft_radius(h):
    return 0.35 if h < 8 else 0.45 if h < 25 else 0.55 if h < 60 else 0.65


def build_supports(contacts, occ, lo, floor_z, tip_r):
    """Supports from each contact point down to the floor (plate/raft top)
    or onto the model below it. Returns manifolds and feet positions."""
    parts, feet, shafts = [], [], []
    for c in contacts:
        i, j = int((c[0] - lo[0]) / XY), int((c[1] - lo[1]) / XY)
        k_top = int((c[2] - 1.6) / DZ)                 # leave room for the tip
        col = occ[i, j, :max(k_top, 0)]
        solid_below = np.nonzero(col)[0]
        tip_len = 1.6
        top = c[2] - 0.15                              # bite into the model
        if len(solid_below):                           # lands on the model
            bottom = lo[2] + (solid_below[-1] + 1) * DZ + 0.15
            on_model = True
        else:
            bottom = floor_z
            on_model = False
        h = top - bottom
        if h < 2.0:
            continue
        r = shaft_radius(h)
        tl = min(tip_len, h / 3)
        parts.append(mf.Manifold.cylinder(tl, r, tip_r, 12).translate((c[0], c[1], top - tl)))
        if on_model:
            parts.append(mf.Manifold.cylinder(tl, tip_r, r, 12).translate((c[0], c[1], bottom)))
            s0, s1 = bottom + tl, top - tl
        else:
            foot_h = min(1.0, h / 4)
            parts.append(mf.Manifold.cylinder(foot_h, r * 2.6, r, 16).translate(
                (c[0], c[1], bottom)))
            feet.append((c[0], c[1], r * 2.6))
            s0, s1 = bottom + foot_h, top - tl
            shafts.append((c[0], c[1], s0, s1, r))
        if s1 > s0:
            parts.append(mf.Manifold.cylinder(s1 - s0, r, r, 12).translate((c[0], c[1], s0)))
    # cross braces between tall plate shafts that stand close together
    braces = 0
    if shafts:
        xy = np.array([[s[0], s[1]] for s in shafts])
        from scipy.spatial import cKDTree
        # each tall shaft is tied to its two nearest tall neighbours
        tall = [i for i, s_ in enumerate(shafts) if s_[3] - s_[2] > 20]
        pairs = set()
        if len(tall) > 2:
            sub = cKDTree(xy[tall])
            _, nn = sub.query(xy[tall], k=3)
            for a, row in enumerate(nn):
                for b in row[1:]:
                    pairs.add((min(tall[a], tall[b]), max(tall[a], tall[b])))
        for a, b in pairs:
            sa, sb = shafts[a], shafts[b]
            if np.hypot(sa[0] - sb[0], sa[1] - sb[1]) > 10:
                continue
            z0, z1 = max(sa[2], sb[2]) + 4, min(sa[3], sb[3]) - 4
            for z in np.arange(z0, z1 - 6, 15.0):
                p, q = np.array([sa[0], sa[1], z]), np.array([sb[0], sb[1], z + 6])
                parts.append(rod(p, q, 0.35))
                braces += 1
    return parts, feet, braces


def rod(p, q, r):
    v = q - p
    L = np.linalg.norm(v)
    z = np.array([0, 0, 1.0])
    axis = np.cross(z, v / L)
    ang = np.degrees(np.arccos(np.clip(v[2] / L, -1, 1)))
    m = mf.Manifold.cylinder(L, r, r, 8)
    if np.linalg.norm(axis) > 1e-9:
        R = rotation(axis, ang)[:3, :3]
        m = m.transform(np.hstack([R, np.zeros((3, 1))]))
    return m.translate(tuple(p))


def raft(feet, z0=0.0):
    """A chamfered raft slab under all the feet."""
    pts = []
    for x, y, r in feet:
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            pts.append([x + (r + 1.5) * np.cos(a), y + (r + 1.5) * np.sin(a)])
    hull = mf.CrossSection([np.array(pts)]).hull()
    base = mf.Manifold.extrude(hull, RAFT_T)
    return base.translate((0, 0, z0))


# ------------------------------------------------------------------- parts

def prepare(mesh, name, outdir, on_raft, tip_r):
    t0 = time.time()
    m = to_manifold(mesh)
    b = np.array(m.bounding_box()).reshape(2, 3)
    floor_z = RAFT_T if on_raft else 0.0
    lift = (RAFT_T + LIFT) if on_raft else 0.0
    m = m.translate((-(b[0, 0] + b[1, 0]) / 2, -(b[0, 1] + b[1, 1]) / 2, lift - b[0, 2]))
    lo, shape = grid_for(m)
    occ = voxelize(m, lo, shape)
    print(f"{name}: sliced {shape[2]} layers in {time.time() - t0:.0f}s")
    contacts, islands = plan_contacts(occ, lo, min_z=0.0 if not on_raft else lift)
    if on_raft:
        # the lowest layers of a lifted part all need support too
        base = np.argwhere(occ[:, :, int(lift / DZ) + 1])
        cell = max(1, int(SPACING / XY))
        _, uniq = np.unique(base // cell, axis=0, return_index=True)
        extra = lo + np.column_stack([(base[uniq, 0] + 0.5) * XY, (base[uniq, 1] + 0.5) * XY,
                                      np.full(len(uniq), int(lift / DZ) * DZ)])
        contacts = np.vstack([contacts, extra]) if len(contacts) else extra
    print(f"{name}: {islands} unsupported regions, {len(contacts)} contact points")
    parts, feet, braces = build_supports(contacts, occ, lo, floor_z, tip_r)
    if on_raft and feet:
        parts.append(raft(feet))
    sup = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    full = mf.Manifold.batch_boolean([m, sup], mf.OpType.Add)
    print(f"{name}: {len(parts)} support pieces, {braces} braces, "
          f"model {m.volume() / 1000:.1f} mL + supports {(full.volume() - m.volume()) / 1000:.1f} mL"
          f" ({time.time() - t0:.0f}s)")

    # verify: re-slice everything and look for layers that start in mid-air
    lo2, shape2 = grid_for(full)
    occ2 = voxelize(full, lo2, shape2)
    # true islands: components with nothing at all in the layer below
    true_islands = 0
    for k in range(1, shape2[2]):
        lab2, n2 = ndimage.label(occ2[:, :, k])
        if n2:
            hit = ndimage.maximum(occ2[:, :, k - 1], lab2, np.arange(1, n2 + 1))
            true_islands += int((hit == 0).sum())
    print(f"{name}: verify -> {true_islands} floating islands")
    if true_islands:
        where = []
        for k in range(1, shape2[2]):
            lab2, n2 = ndimage.label(occ2[:, :, k])
            if n2:
                hit = ndimage.maximum(occ2[:, :, k - 1], lab2, np.arange(1, n2 + 1))
                for c in np.nonzero(hit == 0)[0]:
                    area = (lab2 == c + 1).sum() * XY * XY
                    where.append((round(k * DZ, 1), round(area, 2)))
        print(f"  islands (z mm, area mm2): {where[:15]}")
    tm = to_trimesh(full)
    out = os.path.join(outdir, f"lattice_plumeria_{name}_supported.stl")
    tm.export(out)
    ext = tm.extents
    print(f"{name}: wrote {out}  size {np.round(ext, 1)} mm, watertight={tm.is_watertight}")
    return full, true_islands


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", required=True)
    ap.add_argument("--bloom", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--bloom-tilt", type=float, default=30.0)
    args = ap.parse_args()

    body = trimesh.load(args.body)
    full_body, _ = prepare(body, "body", args.outdir, on_raft=False, tip_r=0.3)

    bloom = trimesh.load(args.bloom)
    bloom.apply_transform(rotation([1, 0, 0], args.bloom_tilt))
    full_bloom, _ = prepare(bloom, "bloom", args.outdir, on_raft=True, tip_r=0.25)

    # both parts on one plate, side by side
    bb = np.array(full_body.bounding_box()).reshape(2, 3)
    bl = np.array(full_bloom.bounding_box()).reshape(2, 3)
    gap = 6.0
    dx_body = -(bb[1, 0] - bb[0, 0] + bl[1, 0] - bl[0, 0] + gap) / 2 - bb[0, 0]
    dx_bloom = dx_body + bb[1, 0] + gap - bl[0, 0]
    plate = mf.Manifold.batch_boolean([full_body.translate((dx_body, 0, 0)),
                                       full_bloom.translate((dx_bloom, 0, 0))], mf.OpType.Add)
    pb = np.array(plate.bounding_box()).reshape(2, 3)
    size = pb[1] - pb[0]
    fits = size[0] <= PLATE_X and size[1] <= PLATE_Y and size[2] <= PLATE_Z
    out = os.path.join(args.outdir, "lattice_plumeria_plate.stl")
    to_trimesh(plate).export(out)
    print(f"plate: {np.round(size, 1)} mm on a {PLATE_X} x {PLATE_Y} x {PLATE_Z} mm "
          f"volume -> {'fits' if fits else 'DOES NOT FIT'}; wrote {out}")


if __name__ == "__main__":
    main()
