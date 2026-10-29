#!/usr/bin/env python3
"""Final build of the ʻōhiʻa lehua bonsai: the tree meshed at 0.12 mm with
the five blossoms and the ʻiʻiwi re-meshed at 0.06 mm and joined in, then a
printer-resolution island check; repeat with extra stems until clean, then
a lighter copy for CHITUBOX (also checked), and the pot.

Each fine box is rebuilt on its own, keeping every piece in it (leaves
crossing a box are separate pieces there and must not be dropped). The
coarse tree is cut 1 mm inside the box and the fine part unioned in, so
they overlap and fuse.

Each round runs in its own process (a round needs about 10 GB).

Usage
  python3 final_build.py OUTDIR            # everything
  python3 final_build.py OUTDIR --round    # one round (used internally)
"""

import gc
import json
import os
import pickle
import subprocess
import sys

import manifold3d as mf
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "gyroid-pendant"))
import ohia_bonsai as ob  # noqa: E402
import lattice_plumeria as lp  # noqa: E402
from bonsai_hibiscus import clear_stems  # noqa: E402
from fractal_plumeria import mesh_volume  # noqa: E402
from gyroid_pendant import verify  # noqa: E402
from plumeria_ring import decimate  # noqa: E402


def to_mf(m):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                               tri_verts=np.asarray(m.faces, np.uint32)))


def stl_welds(m):
    """Edges that would not be shared by exactly two faces once the mesh is
    saved as STL (float32, no topology) and welded again on loading."""
    _, inv = np.unique(m.vertices.astype(np.float32), axis=0, return_inverse=True)
    f = inv.ravel()[m.faces]
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c != 2).sum())


def separate_twins(m):
    """Where thin parts touch, Manifold can leave distinct vertices at the
    same spot (to STL precision); they would fuse on reload and the file
    would no longer be watertight. Push each twin a different few um
    along its own normal until none is left."""
    v = m.vertices.copy()
    for _ in range(4):
        _, inv, counts = np.unique(v.astype(np.float32), axis=0,
                                   return_inverse=True, return_counts=True)
        inv = inv.ravel()
        twins = np.nonzero(counts[inv] > 1)[0]
        if not len(twins):
            break
        order = twins[np.argsort(inv[twins], kind="stable")]
        g = inv[order]
        rank = np.arange(len(order)) - np.searchsorted(g, g)
        n = trimesh.Trimesh(v, m.faces, process=False).vertex_normals[order]
        v[order] -= (0.001 * (1 + rank))[:, None] * n
    return trimesh.Trimesh(v, m.faces, process=False)


def from_mf(M):
    o = M.to_mesh()
    m = trimesh.Trimesh(o.vert_properties[:, :3].astype(np.float64), o.tri_verts, process=False)
    return separate_twins(m)


def fine_part(stems, lo, hi, voxel=0.06):
    """Mesh one box finely, keeping every piece in it."""
    from skimage.measure import marching_cubes
    v, o = ob.build_tree(voxel, stems, quiet=True, region=(lo, hi))
    v = np.pad(v, 1, constant_values=1.0)                 # cap the box walls
    v[v == 0] = 1e-6
    vt, fc, _, _ = marching_cubes(v, 0.0, spacing=(voxel,) * 3)
    del v
    m = trimesh.Trimesh(vt + np.asarray(o) - voxel, fc, process=False)
    keep = [q for q in m.split(only_watertight=False) if len(q.faces) >= 200]
    return mf.Manifold.compose([to_mf(q) for q in keep]).simplify(0.004)


def islands(mesh):
    return [i for i in verify(mesh, 0.05, 0.05)[0] if i[0] > lp.BASE_CUT + 0.15]


def one_round(out):
    """Add stems for the last round's islands, rebuild, check."""
    sp, ip = os.path.join(out, "stems.pkl"), os.path.join(out, "islands.json")
    stems = pickle.load(open(sp, "rb"))
    isl = json.load(open(ip)) if os.path.exists(ip) else []
    if isl:
        cv, co = ob.build_tree(0.3, stems, quiet=True)
        stems += clear_stems([np.array([x, y, z]) for z, a, x, y in isl], cv, 0.3, co)
        del cv
        pickle.dump(stems, open(sp, "wb"))
    vol, o = ob.build_tree(0.12, stems, quiet=True)
    body = to_mf(decimate(mesh_volume(vol, 0.12, o), 0.006))
    del vol
    gc.collect()
    for lo, hi in ob.fine_regions():
        lo, hi = np.asarray(lo) - 1.0, np.asarray(hi) + 1.0
        part = fine_part(stems, lo, hi)
        cut = mf.Manifold.cube((hi - lo - 2.0).tolist()).translate((lo + 1.0).tolist())
        body = (body - cut) + part
        gc.collect()
    full = from_mf(body)
    full = max(full.split(only_watertight=False), key=lambda p: len(p.faces))
    full.export(os.path.join(out, "ohia_bonsai_tree.stl"))
    np.savez(os.path.join(out, "tree.npz"), v=full.vertices, f=full.faces)   # exact topology
    new = islands(full)
    json.dump([[float(a) for a in i] for i in new], open(ip, "w"))
    print(f"round: {len(full.faces)} faces, watertight={full.is_watertight}, "
          f"STL weld faults={stl_welds(full)}, volume={full.volume / 1000:.2f} mL, size={np.round(full.extents, 1)} mm, "
          f"{len(stems)} stems, islands {len(new)} {new[:8]}", flush=True)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    if "--round" in sys.argv:
        one_round(out)
        return
    sp, ip = os.path.join(out, "stems.pkl"), os.path.join(out, "islands.json")
    if not os.path.exists(sp):
        pickle.dump(ob.plan_stems(), open(sp, "wb"))
    for _ in range(6):
        subprocess.run([sys.executable, __file__, out, "--round"], check=True)
        if not json.load(open(ip)):
            break
    else:
        sys.exit("islands remain after 6 rounds")
    z = np.load(os.path.join(out, "tree.npz"))
    full = trimesh.Trimesh(z["v"], z["f"], process=False)
    for tol in (0.012, 0.009, 0.006):
        lite = separate_twins(decimate(full, tol))
        if not islands(lite) and not stl_welds(lite):
            lite.export(os.path.join(out, "ohia_bonsai_tree_lite.stl"))
            print(f"lite: {len(lite.faces)} faces at {tol} mm, no islands", flush=True)
            break
    vol, o = ob.build_pot(0.12)
    pot = separate_twins(decimate(mesh_volume(vol, 0.12, o), 0.006))
    pot.export(os.path.join(out, "ohia_bonsai_pot.stl"))
    print(f"pot: {len(pot.faces)} faces, watertight={pot.is_watertight}, STL weld faults={stl_welds(pot)}, "
          f"islands {len(islands(pot))}", flush=True)
    json.dump({"stems": len(pickle.load(open(sp, "rb")))}, open(os.path.join(out, "build.json"), "w"))


if __name__ == "__main__":
    main()
