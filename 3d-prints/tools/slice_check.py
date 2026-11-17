#!/usr/bin/env python3
"""Printer-style slice check of an STL: slice it layer by layer (as the
printer will, after orienting it) and list islands, i.e. pieces of a layer
that touch nothing in the layer below. Each island needs a support (or a
design fix); the first layers on the plate are not counted.

  python3 slice_check.py part.stl                      # as oriented in the file (z up)
  python3 slice_check.py part.stl --stand --lean 0 30 45
      --stand   stand a flat piece up: its +y (e.g. a hanging loop) goes up
      --lean    then lean it back by these angles (degrees from vertical),
                so its +z face looks up, away from the build plate
  --layer 0.05 --xy 0.05     layer height and raster pitch, mm (smaller is
                             slower and stricter; 0.05/0.025 for jewelry,
                             0.1/0.1 for big flat pieces like suncatchers,
                             whose leaned slices are large)
  --simplify 0.01            for the check only, first simplify meshes over
                             400k faces within this tolerance (mm), since
                             slicing time grows with the face count; 0 = off

Also reports the largest unsupported overhang ledge (beyond 30 degrees) and
the area of downward-facing surface flatter than 30 degrees, which want
supports even when connected.
"""
import argparse
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plumeria-ring"))
from plumeria_ring import decimate  # noqa: E402


def raster(polys, lo, xy, shape):
    """Even-odd fill of slice polygons onto a grid (pixel centres), by
    vectorised scanlines: every edge's crossings with every pixel row,
    sorted per row and filled between pairs."""
    E = []
    for p in polys:
        p = (np.asarray(p, float) - lo) / xy - 0.5
        if len(p) >= 3:
            E.append(np.stack([p, np.roll(p, -1, 0)], 1))
    img = np.zeros(shape, bool)
    if not E:
        return img
    E = np.concatenate(E)                                # (n, 2 ends, xy)
    # rows run along the first axis here (x), scanning across y
    a, b = E[:, 0], E[:, 1]
    x0, x1 = a[:, 0], b[:, 0]
    lo_r = np.ceil(np.minimum(x0, x1)).astype(int)
    hi_r = np.ceil(np.maximum(x0, x1)).astype(int)       # rows lo_r .. hi_r-1 cross
    n = np.clip(hi_r - lo_r, 0, None)
    keep = n > 0
    a, b, lo_r, n = a[keep], b[keep], lo_r[keep], n[keep]
    idx = np.repeat(np.arange(len(n)), n)
    row = lo_r[idx] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    t = (row - a[idx, 0]) / (b[idx, 0] - a[idx, 0])
    col = a[idx, 1] + t * (b[idx, 1] - a[idx, 1])
    o = np.lexsort((col, row))
    row, col = row[o], col[o]
    # pair crossings within each row: (0,1), (2,3), ...
    start = np.r_[0, np.flatnonzero(np.diff(row)) + 1]
    rank = np.arange(len(row)) - np.repeat(start, np.diff(np.r_[start, len(row)]))
    first = (rank % 2 == 0)
    r0, c0 = row[first], col[first]
    c1 = col[np.flatnonzero(first) + 1]
    ok = (r0 >= 0) & (r0 < shape[0])
    r0, c0, c1 = r0[ok], np.clip(np.ceil(c0), 0, shape[1]).astype(int), np.clip(np.ceil(c1), 0, shape[1]).astype(int)
    diff = np.zeros((shape[0], shape[1] + 1), np.int32)
    np.add.at(diff, (r0, c0), 1)
    np.add.at(diff, (r0, c1), -1)
    return np.cumsum(diff, 1)[:, :-1] > 0


def verify(mesh, layer=0.05, xy=0.05):
    """Slice like a printer. Returns islands [(z, area, x, y)], the largest
    overhang ledge beyond 30 degrees (mm^2) and the downward-facing area
    flatter than 30 degrees (mm^2). Same rules as gyroid_pendant.verify."""
    import manifold3d as mf
    from scipy import ndimage
    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                            tri_verts=np.asarray(mesh.faces, np.uint32)))
    b = mesh.bounds
    lo = b[0, :2] - 0.5
    shape = tuple(np.ceil((b[1, :2] + 0.5 - lo) / xy).astype(int))
    nz = int(np.ceil((b[1, 2] - b[0, 2]) / layer))
    prev, islands, ledge = None, [], 0
    reach = max(1, int(round(layer / np.tan(np.radians(30)) / xy)))
    st = ndimage.generate_binary_structure(2, 1)
    for k in range(nz):
        z = b[0, 2] + (k + 0.5) * layer
        cur = raster(m.slice(z).to_polygons(), lo, xy, shape)
        if prev is not None and k > 2:
            lab, n = ndimage.label(cur)
            if n:
                idx = np.arange(1, n + 1)
                near = ndimage.binary_dilation(prev, st, iterations=2)
                hit = ndimage.maximum(near, lab, idx)
                miss = idx[hit == 0]
                if len(miss):
                    area = ndimage.sum(cur, lab, miss) * xy * xy
                    cen = ndimage.center_of_mass(cur, lab, miss)
                    for ar, c in zip(area, cen):
                        if ar >= 0.01:
                            islands.append((round(z, 2), round(float(ar), 3),
                                            round(lo[0] + c[0] * xy, 2), round(lo[1] + c[1] * xy, 2)))
            grown = ndimage.binary_dilation(prev, st, iterations=reach)
            ledge = max(ledge, int((cur & ~grown).sum()))
        prev = cur
    nrm = mesh.face_normals
    flat_down = mesh.area_faces[(nrm[:, 2] < -np.cos(np.radians(30))) &
                                (mesh.triangles_center[:, 2] > b[0, 2] + 2.0)].sum()
    return islands, ledge * xy * xy, flat_down, mesh.area


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stl")
    ap.add_argument("--stand", action="store_true")
    ap.add_argument("--lean", type=float, nargs="*", default=[0.0])
    ap.add_argument("--layer", type=float, default=0.05)
    ap.add_argument("--xy", type=float, default=0.05)
    ap.add_argument("--simplify", type=float, default=0.01)
    a = ap.parse_args()
    m0 = trimesh.load(a.stl)
    print(f"{a.stl}: {len(m0.faces)} faces, watertight={m0.is_watertight}, "
          f"bodies={len(m0.split(only_watertight=False))}, size={np.round(m0.extents, 1)} mm", flush=True)
    if a.simplify and len(m0.faces) > 400_000:
        m0 = decimate(m0, a.simplify)
        print(f"  simplified to {len(m0.faces)} faces (within {a.simplify} mm) for the check", flush=True)
    for lean in (a.lean if a.stand else [None]):
        m = m0.copy()
        if a.stand:
            ang = np.radians(90 - lean)          # +y -> up, +z face tipped up by `lean`
            m.apply_transform(trimesh.transformations.rotation_matrix(ang, [1, 0, 0]))
        m.apply_translation([0, 0, -m.bounds[0, 2]])
        isl, ledge, flat_down, _ = verify(m, a.layer, a.xy)
        tag = "as oriented" if lean is None else f"stood up, leaning {lean:g} deg"
        print(f"  {tag}: height {m.extents[2]:.1f} mm, islands {len(isl)}, "
              f"largest overhang ledge {ledge:.2f} mm^2, flat downward area {flat_down:.0f} mm^2", flush=True)
        for i in isl[:20]:
            print(f"    island at z={i[0]} mm, area {i[1]} mm^2, x={i[2]}, y={i[3]}")


if __name__ == "__main__":
    main()
