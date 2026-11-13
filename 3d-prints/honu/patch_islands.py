#!/usr/bin/env python3
"""Pin islands the final check found, without rebuilding the tiles.

Each island gets the shortest vertical post (r 0.35 mm) that ties it to the
solid below: the post grows downward until a local crop re-checks clean, so
no long needles hang under the turtle. Adapted from silversword/patch_islands.py.

Usage: python3 patch_islands.py OUTDIR
"""
import importlib.util
import json
import os
import sys

import manifold3d as mf
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("honu_final_build", os.path.join(HERE, "final_build.py"))
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)

PIN_R, CROP = 0.35, 5.0
LENGTHS = (0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0)


def local(M, x, y, zi):
    """Islands in a crop around (x, y), ignoring the crop's side walls."""
    lo, hi = np.array([x - CROP, y - CROP, -1.0]), np.array([x + CROP, y + CROP, zi + 3.0])
    c = fb.from_mf(M ^ mf.Manifold.cube((hi - lo).tolist()).translate(lo.tolist()))
    return [i for i in fb.islands(c)
            if abs(i[2] - x) < CROP - 0.6 and abs(i[3] - y) < CROP - 0.6 and i[0] >= zi - 0.3]


def main():
    out = sys.argv[1]
    isl = json.load(open(os.path.join(out, "islands.json")))
    z = np.load(os.path.join(out, "honu.npz"))
    M = fb.to_mf(trimesh.Trimesh(z["v"], z["f"], process=False))
    del z
    pins = []
    for zi, a, x, y in isl:
        if not local(M, x, y, zi):
            print(f"island z={zi:.2f} at ({x:.1f}, {y:.1f}): already tied by an earlier pin", flush=True)
            continue
        for L in LENGTHS:
            pin = mf.Manifold.cylinder(L + 0.4, PIN_R, PIN_R, 24).translate([x, y, zi - L])
            left = local(M + pin, x, y, zi)
            if not left:
                M = M + pin
                pins.append([zi, x, y, L])
                print(f"island z={zi:.2f} at ({x:.1f}, {y:.1f}): pinned with a {L} mm post", flush=True)
                break
        else:
            sys.exit(f"island z={zi:.2f} at ({x:.1f}, {y:.1f}) needs more than {LENGTHS[-1]} mm: {left}")
    M = max(M.decompose(), key=lambda p: p.num_tri())       # cheaper than a trimesh split
    full = fb.from_mf(M)
    del M
    full = fb.clean_export(full, os.path.join(out, "honu.stl"))
    np.savez(os.path.join(out, "honu.npz"), v=full.vertices, f=full.faces)
    print(f"patched: {len(full.faces)} faces, watertight={full.is_watertight}, "
          f"STL weld faults={fb.stl_welds(full)}, volume={full.volume / 1000:.2f} mL", flush=True)
    json.dump([], open(os.path.join(out, "islands.json"), "w"))
    json.dump(pins, open(os.path.join(out, "pinned.json"), "w"))


if __name__ == "__main__":
    main()
