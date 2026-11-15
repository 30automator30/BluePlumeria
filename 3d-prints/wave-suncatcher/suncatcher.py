#!/usr/bin/env python3
"""Wave suncatcher (clear resin, Saturn 4 Ultra), 100 mm across.

A round frame with a hanging loop; inside it a curling wave (the spiral
barrel from nalu/) breaking to the left. Light makes the picture, so it is
drawn in thickness:

  frame     a rounded ring, 5 mm wide, 4.5 mm thick, with a loop at the top
  pane      a 1.4 mm clear pane filling the frame: the sky and the tube
  water     the wave and the sea, 3.6 mm thick, domed toward the front, with
            swirl lines cut in parallel to the wave's face (like the rings
            in a glassy curl) and a torn, ragged lip
  plumeria  a five-petal plumeria cut right through the pane inside the
            barrel, with a low rim round it so its outline glows

The back is flat. Print it standing on edge or tilted 30-45 degrees with
light tree supports on the frame (never on the front face).

Usage
  python3 suncatcher.py --voxel 0.25 --preview
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("fractal-plumeria", "plumeria-ring", "hibiscus-bonsai", "ohia-bonsai", "honu", "nalu"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
from fractal_plumeria import mesh_volume  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402
import nalu  # noqa: E402

D = 100.0                    # outer diameter
RING_W, RING_T = 5.0, 4.5    # frame width and thickness
PANE_T = 1.4                 # the pane: sky and tube
WATER_T = 3.6                # the wave at its thickest
R_OUT = D / 2
R_IN = R_OUT - RING_W
LOOP = dict(r=3.4, tube=1.5)  # hanging loop: hole radius, tube radius
WAVE_H = 66.0                # height of the curl's crest above the sea line
SEA_Y = -30.0                # the sea line (y of the wave's foot)
WAVE_X = -4.0                # where the curl sits across the frame
FLOWER_R = 10.5


def smax(a, b, k):
    return -smin(-a, -b, k)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def wave2d(X, Y):
    """Signed distance in the picture plane to the wave and the sea. The
    nalu barrel profile, drawn with the wave breaking to the left."""
    P = nalu.outline(nalu.BARREL, nalu.BARREL_LIP, 3.0, WAVE_H)   # (y, z) in nalu terms
    sx = 1.25                                                     # stretch it across the frame
    P = np.stack([WAVE_X + P[:, 0] * sx, SEA_Y + P[:, 1]], 1)
    d = nalu.poly_sdf(X, Y, P)
    # the sea: everything below a gently swelling line, both sides of the wave
    sea = Y - (SEA_Y + 1.5 + 1.2 * np.sin(X / 6.0) + 0.6 * np.sin(X / 2.3 + 1.0))
    return smin(d, sea, 2.0)


def petal_poly(R, a, n=40):
    """One obovate petal as a closed polygon, pointing along angle a, offset
    pinwheel-fashion from the centre."""
    t = np.linspace(0, 1, n)
    wid = 0.47 * R * np.minimum(t / 0.62, 1) ** 0.8 * np.clip(1 - (np.maximum(t - 0.62, 0) / 0.38) ** 2.2, 0, 1)
    u = t * R
    left = np.stack([u, wid * 1.1], 1)
    right = np.stack([u, -wid * 0.9], 1)[::-1]
    P = np.vstack([left, right, left[:1]])
    off = 0.15 * R
    c = np.array([-off * np.sin(a), off * np.cos(a)])
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return P @ Rm.T + c


def plumeria2d(X, Y, C, R, rot=0.3):
    """Signed distance to a five-petal plumeria outline: exact per petal
    (polygons), unioned, with a round centre."""
    d = np.hypot(X - C[0], Y - C[1]) - 0.22 * R
    for k in range(5):
        P = petal_poly(R, 2 * np.pi * k / 5 + rot) + C
        d = np.minimum(d, nalu.poly_sdf(X, Y, P))
    return d


def tube_centre():
    """The middle of the barrel: of the open points under the lip (water
    straight above them), the one farthest from the water."""
    xs = np.arange(-R_IN, R_IN, 0.25)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    dw = wave2d(X, Y)
    solid = dw < 0
    above = np.flip(np.maximum.accumulate(np.flip(solid, 1), 1), 1)   # water somewhere above
    cand = (dw > 0) & above & (np.hypot(X, Y) < R_IN - 3) & (Y > SEA_Y + 6)
    i = np.argmax(np.where(cand, dw, -1))
    return np.array([X.ravel()[i], Y.ravel()[i]]), dw.ravel()[i]


def build(voxel):
    t0 = time.time()
    lo = np.array([-R_OUT - 1, -R_OUT - 1, -0.5])
    hi = np.array([R_OUT + 1, R_OUT + 2 * LOOP["r"] + 2 * LOOP["tube"] + 2, RING_T + 0.5])
    xs = np.arange(lo[0], hi[0], voxel, dtype=np.float32)
    ys = np.arange(lo[1], hi[1], voxel, dtype=np.float32)
    zs = np.arange(lo[2], hi[2], voxel, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    r = np.hypot(X, Y)
    inside = R_IN + 0.5 - r                                      # >0 inside the frame's inner edge

    # the picture, in 2D
    dw = wave2d(X, Y)
    dw = np.maximum(dw, -inside)                                  # clipped to the frame's opening
    # front heights (z of the front face) over the picture plane
    depth = np.clip(-dw / 3.0, 0, 1)
    water_top = PANE_T + (WATER_T - PANE_T) * sstep(0, 1, depth) ** 0.7
    # swirl lines: a spiral winding out from the barrel through the wave,
    # like the rings in a glassy curl, plus one line tracing the face; the
    # sea keeps its own level ripples
    C, room = tube_centre()
    rr = np.hypot(X - C[0], Y - C[1])
    th = np.arctan2(Y - C[1], X - C[0])
    spacing = 2.3 + 0.03 * rr
    spiral = np.abs(np.sin(np.pi * (rr / spacing + th / (2 * np.pi)))) ** 8
    body = sstep(SEA_Y + 1.0, SEA_Y + 5.0, Y)
    trace = np.exp(-((-dw - 1.3) / 0.35) ** 2)
    ripple = np.abs(np.sin(np.pi * (Y - SEA_Y + 0.8 * np.sin(X / 5.0)) / 2.4)) ** 8 * (1 - body)
    groove = np.maximum(np.maximum(spiral * body, trace), ripple)
    water_top = water_top - 0.45 * groove * sstep(0.6, 1.6, -dw)
    top = np.where(dw < 0, water_top, PANE_T)
    dflower = plumeria2d(X, Y, C, min(FLOWER_R, room - 1.6))
    # the plumeria: a rim round it on the pane, then cut right through
    rim = np.clip(1 - np.abs(dflower - 0.5) / 0.55, 0, 1)
    top = np.maximum(top, np.where(dw > 0.3, PANE_T + 0.9 * sstep(0, 1, rim * 1.6), 0))
    # the frame: a rounded ring
    ring_r = np.abs(r - (R_OUT - RING_W / 2)) - RING_W / 2
    say = print
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    for k, z in enumerate(zs):
        picture = np.maximum(z - top, -z)                       # slab from the back (z=0) up to the front
        picture = np.maximum(picture, -inside + 0.5)
        frame = smax(ring_r, np.abs(z - RING_T / 2) - RING_T / 2, 1.4)
        d = smin(picture, frame, 0.8)
        d = np.maximum(d, -dflower + 0.0 * z)                    # the cut-out
        vol[:, :, k] = d
    # hanging loop: a torus in the picture plane, standing on the frame
    lc = np.array([0.0, R_OUT + LOOP["r"] + LOOP["tube"] * 0.4, RING_T / 2])
    for k, z in enumerate(zs):
        q = np.hypot(X - lc[0], Y - lc[1]) - LOOP["r"] - LOOP["tube"]
        tor = np.hypot(q, z - lc[2]) - LOOP["tube"] * 1.1
        vol[:, :, k] = smin(vol[:, :, k], tor, 0.8)
    say(f"  built {time.time() - t0:.0f}s")
    return vol, (xs[0], ys[0], zs[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.25)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    t0 = time.time()
    vol, origin = build(args.voxel)
    mesh = mesh_volume(vol, args.voxel, origin)
    if not args.preview:
        mesh = decimate(mesh, 0.004)
    print(f"suncatcher: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, {time.time() - t0:.0f}s")
    out = os.path.join(args.outdir, "wave_suncatcher.stl")
    mesh.export(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
