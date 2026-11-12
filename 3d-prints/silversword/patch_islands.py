#!/usr/bin/env python3
"""Pin islands the final check found that sit just above the plinth (needle
tips hovering over the cinders), without rebuilding the tiles: a short
post from each island straight down into the plinth is unioned into the
finished mesh, and each spot is re-checked on a local crop.

Usage: python3 patch_islands.py OUTDIR
"""
import json
import os
import sys

import importlib.util

import manifold3d as mf
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("sw_final_build", os.path.join(HERE, "final_build.py"))
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)

PIN_R, CROP = 0.35, 6.0


def local(M, x, y, ztop):
    """Islands in a crop around (x, y), ignoring the crop's side walls."""
    lo, hi = np.array([x - CROP, y - CROP, -1.0]), np.array([x + CROP, y + CROP, ztop])
    c = fb.from_mf(M ^ mf.Manifold.cube((hi - lo).tolist()).translate(lo.tolist()))
    return [i for i in fb.islands(c) if abs(i[2] - x) < CROP - 0.6 and abs(i[3] - y) < CROP - 0.6]


def main():
    out = sys.argv[1]
    isl = json.load(open(os.path.join(out, "islands.json")))
    z = np.load(os.path.join(out, "silversword.npz"))
    M = fb.to_mf(trimesh.Trimesh(z["v"], z["f"], process=False))
    del z
    for zi, a, x, y in isl:
        print(f"island z={zi:.2f} at ({x:.1f}, {y:.1f}): before {local(M, x, y, zi + 4)}", flush=True)
        pin = mf.Manifold.cylinder(zi + 0.4, PIN_R, PIN_R, 24).translate([x, y, 0.0])
        M = M + pin
        print(f"  after {local(M, x, y, zi + 4)}", flush=True)
    full = fb.from_mf(M)
    full = max(full.split(only_watertight=False), key=lambda p: len(p.faces))
    print(f"patched: {len(full.faces)} faces, watertight={full.is_watertight}, "
          f"STL weld faults={fb.stl_welds(full)}, volume={full.volume / 1000:.2f} mL", flush=True)
    full.export(os.path.join(out, "silversword.stl"))
    np.savez(os.path.join(out, "silversword.npz"), v=full.vertices, f=full.faces)
    json.dump([], open(os.path.join(out, "islands.json"), "w"))
    json.dump(isl, open(os.path.join(out, "pinned.json"), "w"))


if __name__ == "__main__":
    main()
