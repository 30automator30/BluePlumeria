#!/usr/bin/env python3
"""Final build of the honu over coral: the whole piece meshed at 0.2 mm, then
everything above the plate re-meshed at 0.08 mm in tiles and joined in
(tiles with no surface in them are skipped); then a printer-resolution
island check, repeated with extra stems until clean; then a lighter copy
(also checked).

Each tile is rebuilt on its own and saved, so a restarted build picks up
where it stopped. Every piece in a tile is kept: needles crossing it are
separate pieces there. The coarse model is cut 1 mm inside each tile and
the fine part unioned in, so they overlap and fuse. Twin vertices are
pushed apart so the STL reloads watertight.

Usage
  python3 final_build.py OUTDIR            # everything
  python3 final_build.py OUTDIR --round    # one round (used internally)
  python3 final_build.py OUTDIR --lite     # only the lighter copy (after patch_islands.py)
"""

import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys

import importlib.util

import manifold3d as mf
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "gyroid-pendant"))
import honu as sw  # noqa: E402
import lattice_plumeria as lp  # noqa: E402
from bonsai_hibiscus import clear_stems  # noqa: E402
from fractal_plumeria import mesh_volume  # noqa: E402
from plumeria_ring import decimate  # noqa: E402

# STL-safe mesh helpers and the printer-resolution check, shared with the ʻōhiʻa build
_spec = importlib.util.spec_from_file_location("ohia_final_build", os.path.join(HERE, "..", "ohia-bonsai", "final_build.py"))
_ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ob)
from_mf, islands, separate_twins, stl_welds, to_mf = _ob.from_mf, _ob.islands, _ob.separate_twins, _ob.stl_welds, _ob.to_mf

BODY_VOXEL, FINE_VOXEL, TILE = 0.2, 0.08, 34.0
if os.environ.get("HONU_DRY"):            # a quick coarse run of the whole pipeline, to test it
    BODY_VOXEL, FINE_VOXEL = 0.4, 0.25
TILE_TOL = 0.012   # tiles simplified on joining: well under the printer's ~19 um pixel, ~4x fewer faces


def clean_export(mesh, path):
    """Save as STL the way a slicer will read it back: export, reload (which
    welds vertices), drop zero-volume slivers left on tile seams (two
    back-to-back triangles), and save again. Returns the reloaded mesh."""
    mesh.export(path)
    r = trimesh.load(path)
    parts = sorted(r.split(only_watertight=False), key=lambda p: -len(p.faces))
    extra = [abs(p.volume) for p in parts[1:]]
    if extra and max(extra) > 1e-3:
        raise RuntimeError(f"loose pieces with volume {max(extra):.4f} mm^3 after reload")
    r = parts[0]
    r.export(path)
    r = trimesh.load(path)
    if not r.is_watertight:
        raise RuntimeError("STL does not reload watertight")
    return r


def regions():
    """Boxes to re-mesh finely: the reef and sea floor, and the honu."""
    _, (tlo, thi), _ = sw.honu(sw.TURTLE["C"], sw.TURTLE["yaw"], sw.TURTLE["pitch"], sw.TURTLE["roll"], sw.CONTACT)
    reef = (np.array([-sw.BASE[0] - 2.5, -sw.BASE[1] - 2.5, 0.0]), np.array([sw.BASE[0] + 2.5, sw.BASE[1] + 2.5, 46.0]))
    turtle = (np.array([tlo[0], tlo[1], 46.0]), np.asarray(thi, float))
    return [reef, turtle]


def has_surface(stems, lo, hi):
    """A cheap coarse look: does any surface pass through this box?"""
    v, _ = sw.build(0.5, stems, quiet=True, region=(lo, hi))
    return bool((v < 0.4).any() and (v > -0.4).any())


def tiles():
    out = []
    for lo, hi in regions():
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        n = np.maximum(np.ceil((hi - lo) / TILE), 1).astype(int)
        step = (hi - lo) / n
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    a = lo + step * [i, j, k]
                    out.append((a, a + step))
    return out


def fine_part(stems, lo, hi, cache):
    """Mesh one tile finely, keeping every piece in it (cached on disk)."""
    if os.path.exists(cache):
        z = np.load(cache)
        if "empty" in z.files:
            return None
        return mf.Manifold(mf.Mesh(vert_properties=z["v"], tri_verts=z["f"]))
    from skimage.measure import marching_cubes
    v, o = sw.build(FINE_VOXEL, stems, quiet=True, region=(lo, hi))
    v = np.pad(v, 1, constant_values=1.0)                 # cap the box walls
    v[v == 0] = 1e-6
    vt, fc, _, _ = marching_cubes(v, 0.0, spacing=(FINE_VOXEL,) * 3)
    del v
    m = trimesh.Trimesh(vt + np.asarray(o) - FINE_VOXEL, fc, process=False)
    keep = [q for q in m.split(only_watertight=False) if len(q.faces) >= 20]
    M = mf.Manifold.compose([to_mf(q) for q in keep]).simplify(0.004) if keep else mf.Manifold()
    o2 = M.to_mesh()
    np.savez(cache, v=o2.vert_properties[:, :3].astype(np.float32), f=o2.tri_verts.astype(np.uint32))
    return M


def one_round(out):
    sp, ip = os.path.join(out, "stems.pkl"), os.path.join(out, "islands.json")
    stems = pickle.load(open(sp, "rb"))
    isl = json.load(open(ip)) if os.path.exists(ip) else []
    if isl:
        cv, co = sw.build(0.3, stems, quiet=True)
        stems += clear_stems([np.array([x, y, z]) for z, a, x, y in isl], cv, 0.3, co)
        del cv
        pickle.dump(stems, open(sp, "wb"))
        if os.path.exists(ip):
            os.remove(ip)
    key = hashlib.md5(pickle.dumps([(np.round(p, 3), np.round(r, 3)) for p, r in stems])).hexdigest()[:8]
    tdir = os.path.join(out, "tiles_" + key)
    os.makedirs(tdir, exist_ok=True)
    tl = tiles()
    for n_ in range(len(tl)):                             # each tile in its own process, to bound memory
        if not os.path.exists(os.path.join(tdir, f"t{n_:03d}.npz")):
            subprocess.run([sys.executable, __file__, out, "--tile", str(n_), tdir], check=True)
    vol, o = sw.build(BODY_VOXEL, stems, quiet=True)
    body = to_mf(decimate(mesh_volume(vol, BODY_VOXEL, o), 0.008))
    del vol
    gc.collect()
    for n_, (lo, hi) in enumerate(tl):
        lo, hi = lo - 1.0, hi + 1.0
        part = fine_part(stems, lo, hi, os.path.join(tdir, f"t{n_:03d}.npz"))
        if part is None:                                  # nothing here
            continue
        part = part.simplify(TILE_TOL)
        cut = mf.Manifold.cube((hi - lo - 2.0).tolist()).translate((lo + 1.0).tolist())
        body = (body - cut) + part
        body.num_tri()                                    # evaluate now, so the CSG tree does not pile up
        print(f"  tile {n_ + 1}/{len(tl)}", flush=True)
        gc.collect()
    full = from_mf(body)
    full = max(full.split(only_watertight=False), key=lambda p: len(p.faces))
    full = clean_export(full, os.path.join(out, "honu.stl"))
    np.savez(os.path.join(out, "honu.npz"), v=full.vertices, f=full.faces)
    new = islands(full)
    json.dump([[float(a) for a in i] for i in new], open(ip, "w"))
    print(f"round: {len(full.faces)} faces, watertight={full.is_watertight}, STL weld faults={stl_welds(full)}, "
          f"volume={full.volume / 1000:.2f} mL, size={np.round(full.extents, 1)} mm, "
          f"{len(stems)} stems, islands {len(new)} {new[:8]}", flush=True)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    if "--tile" in sys.argv:
        n_, tdir = int(sys.argv[sys.argv.index("--tile") + 1]), sys.argv[sys.argv.index("--tile") + 2]
        lo, hi = tiles()[n_]
        stems = pickle.load(open(os.path.join(out, "stems.pkl"), "rb"))
        cache = os.path.join(tdir, f"t{n_:03d}.npz")
        if not has_surface(stems, lo - 1.0, hi + 1.0):
            np.savez(cache, v=np.zeros((0, 3), np.float32), f=np.zeros((0, 3), np.uint32), empty=1)
            print(f"  tile {n_ + 1} empty", flush=True)
            return
        fine_part(stems, lo - 1.0, hi + 1.0, cache)
        print(f"  built tile {n_ + 1}", flush=True)
        return
    if "--round" in sys.argv:
        one_round(out)
        return
    sp, ip = os.path.join(out, "stems.pkl"), os.path.join(out, "islands.json")
    if not os.path.exists(sp):
        pickle.dump([], open(sp, "wb"))     # the design checks clean; repair rounds add stems if needed
    if "--lite" in sys.argv:                                # only (re)make the lighter copy
        return lite(out)
    for _ in range(6):
        subprocess.run([sys.executable, __file__, out, "--round"], check=True)
        if not json.load(open(ip)):
            break
    else:
        sys.exit("islands remain after 6 rounds")
    lite(out)


def lite(out):
    sp = os.path.join(out, "stems.pkl")
    z = np.load(os.path.join(out, "honu.npz"))
    full = trimesh.Trimesh(z["v"], z["f"], process=False)
    tols = os.environ.get("HONU_LITE_TOLS", "0.025,0.02,0.016")   # e.g. HONU_LITE_TOLS=0.016 to retry one
    for tol in [float(t) for t in tols.split(",")]:
        lt = separate_twins(decimate(full, tol))
        isl, welds = islands(lt), stl_welds(lt)
        if not isl and not welds:
            lt = clean_export(lt, os.path.join(out, "honu_lite.stl"))
            print(f"lite: {len(lt.faces)} faces at {tol} mm, no islands", flush=True)
            break
        print(f"  lite at {tol} mm: {len(isl)} islands {isl[:4]}, {welds} weld faults; trying finer", flush=True)
    json.dump({"stems": len(pickle.load(open(sp, "rb")))}, open(os.path.join(out, "build.json"), "w"))


if __name__ == "__main__":
    main()
