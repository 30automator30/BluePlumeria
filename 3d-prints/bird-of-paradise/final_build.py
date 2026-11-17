#!/usr/bin/env python3
"""Final build of the tropical arrangement: the body meshed at 0.12 mm with
the three bird of paradise heads re-meshed at 0.06 mm and joined in, then a
printer-resolution island check; repeat with extra stems until clean, then a
lighter copy for CHITUBOX (also checked).

Each head is rebuilt in its own box. Everything in the box is kept: leaves
crossing it are separate pieces there, and must not be dropped (an earlier
build cut them off flat at the box walls). The coarse body is cut 1 mm
inside the box and the fine head is unioned in, so they overlap and fuse.

A full run needs about 14 GB and a couple of hours per round; run one
round per process if memory is tight.

Usage
  python3 final_build.py OUTDIR
"""

import gc
import json
import os
import pickle
import sys

import manifold3d as mf
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "gyroid-pendant"))
import bird_of_paradise as b  # noqa: E402
import lattice_plumeria as lp  # noqa: E402
from bonsai_hibiscus import clear_stems  # noqa: E402
from fractal_plumeria import mesh_volume  # noqa: E402
from gyroid_pendant import verify  # noqa: E402
from plumeria_ring import decimate  # noqa: E402


def to_mf(m):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                               tri_verts=np.asarray(m.faces, np.uint32)))


def from_mf(M):
    o = M.to_mesh()
    return trimesh.Trimesh(o.vert_properties[:, :3].astype(np.float64), o.tri_verts, process=False)


def fine_head(stems, lo, hi, voxel=0.06):
    """Mesh one head box finely, keeping every piece in it."""
    from skimage.measure import marching_cubes
    v, o = b.build(voxel, stems, quiet=True, region=(lo, hi))
    v = np.pad(v, 1, constant_values=1.0)                 # cap the box walls
    v[v == 0] = 1e-6
    vt, fc, _, _ = marching_cubes(v, 0.0, spacing=(voxel,) * 3)
    del v
    m = trimesh.Trimesh(vt + np.asarray(o) - voxel, fc, process=False)
    keep = [q for q in m.split(only_watertight=False) if len(q.faces) >= 200]
    return mf.Manifold.compose([to_mf(q) for q in keep]).simplify(0.004)


def build_round(stems):
    vol, o = b.build(0.12, stems, quiet=True)
    body = to_mf(decimate(mesh_volume(vol, 0.12, o), 0.006))
    del vol
    gc.collect()
    for _, _, (lo, hi) in b.parts()[2]:
        lo, hi = np.asarray(lo) - 1.0, np.asarray(hi) + 1.0
        head = fine_head(stems, lo, hi)
        cut = mf.Manifold.cube((hi - lo - 2.0).tolist()).translate((lo + 1.0).tolist())
        body = (body - cut) + head
        gc.collect()
    full = from_mf(body)
    return max(full.split(only_watertight=False), key=lambda p: len(p.faces))


def islands(mesh):
    return [i for i in verify(mesh, 0.05, 0.05)[0] if i[0] > lp.BASE_CUT + 0.15]


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sp = os.path.join(out, "stems.pkl")
    stems = pickle.load(open(sp, "rb")) if os.path.exists(sp) else b.plan_stems()
    for rnd in range(5):
        pickle.dump(stems, open(sp, "wb"))
        full = build_round(stems)
        full.export(os.path.join(out, "bird_of_paradise.stl"))
        isl = islands(full)
        print(f"round {rnd}: {len(full.faces)} faces, watertight={full.is_watertight}, "
              f"islands {len(isl)}", flush=True)
        if not isl:
            break
        cv, co = b.build(0.3, stems, quiet=True)
        stems += clear_stems([np.array([x, y, z]) for z, a, x, y in isl], cv, 0.3, co)
    for tol in (0.012, 0.009, 0.006):
        lite = decimate(full, tol)
        if not islands(lite):
            lite.export(os.path.join(out, "bird_of_paradise_lite.stl"))
            print(f"lite: {len(lite.faces)} faces at {tol} mm, no islands")
            break
    json.dump({"stems": len(stems)}, open(os.path.join(out, "build.json"), "w"))


if __name__ == "__main__":
    main()
