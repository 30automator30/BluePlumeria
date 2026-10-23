#!/usr/bin/env python3
"""Hibiscus sculpture (resin / MSLA, Saturn 4 Ultra): companion to the
lattice plumeria, 1/phi of its height (about 99 mm), printed in one piece
straight on the plate with no supports.

Golden layout (mm):
  flower      an open hibiscus whose petals reach H/phi^2 (37.8) from the
              centre, so it is about 75 mm across. Five broad petals wrap
              like a pinwheel, one edge of each tucked under the next, with
              ruffled margins. Each petal has a bold rim, a solid "eye" at
              the claw (out to 1/phi^2 of its length) and a flowing vein
              lattice: 5 veins branch into 9 on an arc at 1/phi, so every
              vein ends on the eye, the arc or the rim, each joint rounded
              with a smooth fillet.
              A staminal column arches up from the centre, with 13 anthers
              set at the golden angle and 5 stigma pads at its tip.
  trunk       three round strands plaited like a braided hibiscus standard,
              fused where they cross, turning slowly round the trunk and
              gathering into the calyx at H/phi. At the foot each strand
              splays into a golden root curl; side shoots carry a twisted
              bud and a golden curl.

Printability: the base is cut flat to sit on the plate. The petals rise
outward from the calyx (a trumpet that never flattens past ~20 degrees), so
nearly everything grows from below. What would still start in mid-air gets
a question-mark curl or a slim S-curved stem, found by slicing the model
layer by layer (the same planner as the plumeria body).

Usage
  python3 hibiscus_sculpture.py --voxel 0.25 --preview   # quick look
  python3 hibiscus_sculpture.py                          # final STL
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fractal-plumeria"))
sys.path.insert(0, os.path.join(HERE, "..", "plumeria-ring"))
import lattice_plumeria as lp  # noqa: E402
from fractal_plumeria import capsule_chain, mesh_volume  # noqa: E402
from lattice_plumeria import (PHI, RIM_T, RIM_W, STRUT_T, STRUT_W, BASE_CUT,  # noqa: E402
                              add, bezier, frame_from_axis, soft_band, to_local)
from plumeria_ring import decimate, smin  # noqa: E402

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))
H = lp.H / PHI                       # 98.9 overall height
R0 = lp.R0 / PHI                     # 18.4 base radius
NECK_Z = H / PHI ** 2                # 37.8
NECK_R = R0 / PHI ** 3               # 4.3
RF = H / PHI ** 2                    # 37.8 petal reach from the centre
RHO0 = 3.5                           # petals start this far out
HT = 0.55 * RF                       # trumpet rise from claw to tip
TILT = np.radians(10)                # flower leans toward the viewer (-y)
AXIS = np.array([0.0, -np.sin(TILT), np.cos(TILT)])
CENTRE = np.array([0.0, -2.0, H / PHI])          # top of the calyx
FRAME = (CENTRE,) + frame_from_axis(AXIS)
CALYX_DEPTH = 14.0                   # calyx runs this far down the axis
EYE_T = PHI ** -2                    # solid eye out to 0.382 of the petal
FILLET = 1.3                         # radius where veins join each other, rim and eye


# ------------------------------------------------------------ the flower

def trumpet(rho):
    """Petal surface height above the calyx, and its slope."""
    s = np.clip((rho - RHO0) / (RF - RHO0), 1e-3, 1.2)
    return HT * s ** 0.62, 0.62 * HT / (RF - RHO0) * s ** -0.38


def petal_half_width(t, wmax):
    """Obovate: widening to 0.72 of the length, then a rounded tip."""
    t = np.clip(t, 0, 1)
    tip = 0.808 * np.sqrt(np.clip(1 - ((t - 0.72) / 0.28) ** 2, 0, 1))
    return wmax * np.where(t < 0.72, t ** 0.65, tip) + 1e-3


def petal_sdf(lx, ly, lz, i):
    """One hibiscus petal in the flower's local frame (z along the axis)."""
    L = RF - RHO0
    rho = np.sqrt(lx ** 2 + ly ** 2) + 1e-6
    u = rho - RHO0
    t = u / L
    phi0 = 2 * np.pi * i / 5 + 0.35 * np.clip(t, 0, 1)        # pinwheel swirl
    dphi = np.angle(np.exp(1j * (np.arctan2(ly, lx) - phi0)))
    v = rho * dphi
    half = petal_half_width(t, 0.78 * L)
    o = np.maximum(np.abs(v) - half, np.maximum(-u, u - L))  # outline (2D)
    sv = np.clip(v / half, -1, 1)

    h, slope = trumpet(rho)
    h = h + 1.6 * sv * np.clip(t, 0, 1)                       # one edge over the next
    psi = np.arctan2(v, u - 0.45 * L)
    edge = np.clip(1 + o / 6.0, 0, 1)                         # ruffle only the margin
    h = h + 1.1 * np.clip((t - 0.45) / 0.4, 0, 1) * edge * np.sin(11 * psi)
    dz = np.abs(lz - h) / np.sqrt(1 + slope ** 2)

    rim = np.maximum(o, -(o + RIM_W))
    out = soft_band(rim, dz - RIM_T / 2, RIM_W / 2, RIM_T / 2)
    eye = np.maximum(o, u - EYE_T * L)                       # solid eye at the claw
    out = smin(out, soft_band(eye, dz - 0.8, 5.0, 0.8), FILLET)

    # veins: 5 fan veins branch into 9 at 1/phi; arcs at golden fractions
    sf = sv + 0.10 * np.sin(np.pi * 1.5 * np.clip(t, 0, 1)) * (1 - sv ** 2)  # veins flow
    fan5 = np.abs(((sf + 1) * 2.5) % 1.0 - 0.5) / 2.5 * half
    fan9 = np.abs(((sf + 1) * 4.5) % 1.0 - 0.5) / 4.5 * half
    fan = np.where(t < 1 / PHI, fan5, fan9)
    arcs = np.min([np.abs(u - f * L) for f in (EYE_T, PHI ** -1)], axis=0)
    inner = o + RIM_W * 0.6
    # smooth radii where veins meet arcs (in plane), and the rim and eye (below)
    strut = np.maximum(smin(fan, arcs, FILLET) - STRUT_W / 2, inner)
    out = smin(out, soft_band(strut, dz - STRUT_T / 2, STRUT_W / 2, STRUT_T / 2), FILLET)
    return out


def column_parts():
    """Staminal column path, anthers (golden angle) and stigma pads."""
    p = bezier(np.array([0, 0, -1.0]), np.array([0, 0, 12.0]), np.array([3.0, 0, 24.0]),
               np.array([7.0, 0, 33.0]), n=40)
    rad = np.linspace(1.7, 1.15, len(p))
    blobs = []
    for k in range(13):
        s = 0.55 + 0.33 * k / 12
        j = int(s * (len(p) - 1))
        tan = p[min(j + 1, len(p) - 1)] - p[j - 1]
        tan /= np.linalg.norm(tan)
        e1 = np.cross(tan, [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(tan, e1)
        a = k * GOLDEN_ANGLE
        out = np.cos(a) * e1 + np.sin(a) * e2
        blobs.append((p[j], p[j] + (rad[j] + 0.9) * out + 0.5 * tan, 0.35, 0.62))
    tip, tan = p[-1], (p[-1] - p[-4]) / np.linalg.norm(p[-1] - p[-4])
    e1 = np.cross(tan, [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(tan, e1)
    for k in range(5):
        a = 2 * np.pi * k / 5
        end = tip + 2.2 * (0.75 * tan + 0.66 * (np.cos(a) * e1 + np.sin(a) * e2))
        blobs.append((tip, end, 0.45, 0.85))
    return p, rad, blobs


def flower_sdf(x, y, z):
    lx, ly, lz = to_local(FRAME, x, y, z)
    d = np.full(lx.shape, 1e9, np.float32)
    for i in range(5):
        d = smin(d, petal_sdf(lx, ly, lz, i), 0.5)
    # calyx: a five-lobed cup under the petals that tapers into the stem
    th = np.arctan2(ly, lx) + np.pi / 5
    lobe = 4.8 * (1 + 0.16 * np.cos(5 * th) * np.clip((lz + 6) / 6, 0, 1))
    cup = np.sqrt(lx ** 2 + ly ** 2) - lobe * np.sqrt(np.clip(1 - ((lz + 3.0) / 5.5) ** 2, 0, 1))
    cup = np.maximum(cup, np.maximum(lz - 1.5, -lz - 8.5))
    s = np.clip(-lz / CALYX_DEPTH, 0, 1)
    cone = np.maximum(np.sqrt(lx ** 2 + ly ** 2) - (4.6 - 2.0 * s ** 0.8),
                      np.maximum(-lz - CALYX_DEPTH, lz + 1.0))
    d = smin(d, smin(cup, cone, 1.5), 1.0)
    p, rad, blobs = column_parts()
    d = smin(d, capsule_chain(lx, ly, lz, p, rad), 0.8)
    for a, b, r_stalk, r_bead in blobs:
        stalk = capsule_chain(lx, ly, lz, np.array([a, b]), np.array([r_stalk, r_stalk]))
        bead = np.sqrt((lx - b[0]) ** 2 + (ly - b[1]) ** 2 + (lz - b[2]) ** 2) - r_bead
        d = smin(d, smin(stalk, bead, 0.3), 0.3)
    return d


def bud_sdf_fn(base, axis, length=15.0, radius=3.4):
    """A twisted hibiscus bud: a spindle with five spiral furls and sepals."""
    fr = (base,) + frame_from_axis(axis)

    def f(x, y, z):
        lx, ly, lz = to_local(fr, x, y, z)
        s = np.clip(lz / length, 0, 1)
        r = radius * np.sin(np.pi * np.clip(s * 0.95 + 0.05, 0, 1)) ** 0.7 * (1 - 0.35 * s)
        th = np.arctan2(ly, lx) - 4.0 * s                     # the furls twist up
        r = r * (1 - 0.10 * np.abs(np.sin(2.5 * th)))
        body = np.maximum(np.sqrt(lx ** 2 + ly ** 2) - r, np.maximum(-lz, lz - length))
        tip = np.sqrt(lx ** 2 + ly ** 2 + (lz - length) ** 2) - 0.6
        sep = np.sqrt(lx ** 2 + ly ** 2 + ((lz - 2.5) / 1.6) ** 2) - radius * 0.95
        sep = np.maximum(sep, lz - 4.5)
        return smin(smin(body, tip, 1.0), sep, 0.8)
    return f


# ------------------------------------------------------------------- body

def stem_axis(z):
    """Straight up to the neck, then an S to the bottom of the calyx."""
    cb = CENTRE - CALYX_DEPTH * AXIS
    z = np.asarray(z, float)
    s = np.clip((z - NECK_Z) / (cb[2] - NECK_Z), 0, 1)
    x = cb[0] * (3 * s ** 2 - 2 * s ** 3) + 2.5 * np.sin(np.pi * s)
    y = cb[1] * s ** 2
    return np.stack([x, y, z], -1)


BRAID_R = 2.5                        # radius of each trunk strand
BRAID_PITCH = 24.0                   # rise of one full plait cycle (6 crossings)
BUD_AT = np.array([13.0, 9.0, 47.0])     # bud base, beside the trunk
BUD_AXIS = np.array([0.45, 0.25, 0.86])


def braid():
    """Three round strands plaited like a braided hibiscus standard. Each
    weaves side to side and over-under its neighbours (a figure-eight in
    the plait's plane), the plait turns slowly round the trunk, and the
    strands taper and gather into the calyx. They touch at every crossing,
    so the trunk prints as one solid piece."""
    cb = CENTRE - CALYX_DEPTH * AXIS
    z = np.linspace(3.0, cb[2] + 3.0, 260)
    s = (z - z[0]) / (z[-1] - z[0])
    t = 2 * np.pi * (z - z[0]) / BRAID_PITCH
    c = stem_axis(z)
    psi = np.pi * 0.6 * s                              # the plait turns round
    e1 = np.column_stack([np.cos(psi), np.sin(psi), 0 * psi])
    e2 = np.column_stack([-np.sin(psi), np.cos(psi), 0 * psi])
    taper = 1 - 0.2 * s
    gather = np.clip((1 - s) / 0.15, 0, 1) ** 0.7      # close up into the calyx
    strands = []
    for k in range(3):
        ph = 2 * np.pi * k / 3
        wx = 1.6 * BRAID_R * taper * gather * np.sin(t + ph)
        wy = 1.05 * BRAID_R * taper * gather * np.sin(2 * (t + ph))
        strands.append((c + wx[:, None] * e1 + wy[:, None] * e2, BRAID_R * taper))
    return strands


def body_parts():
    """Braided trunk with root curls, a bud shoot and a curl shoot."""
    tubes, extras = [], []
    strands = braid()
    for k, (pts, rad) in enumerate(strands):
        # each strand splays out at the foot into a golden root curl
        out = pts[0] - np.array([0, 0, pts[0][2]])
        out = out / np.linalg.norm(out) if np.linalg.norm(out) > 0.5 else \
            np.array([np.cos(2 * np.pi * k / 3), np.sin(2 * np.pi * k / 3), 0])
        start = pts[0] + 3.0 * out - [0, 0, 1.5]
        root = lp.ground_curl(start, -out, 10.0 + 2.0 * k, 1 if k % 2 else -1)
        path = np.vstack([root, pts])
        rr = np.concatenate([np.linspace(1.0, BRAID_R * 1.15, len(root)), rad])
        tubes.append((path, rr))
    # bud shoot from strand 0, curl shoot from strand 1
    p, rad = strands[0]
    j = int(0.45 * len(p))
    tubes.append((bezier(p[j], p[j] + [3, 1, 4], BUD_AT - 5 * BUD_AXIS, BUD_AT + 0.5 * BUD_AXIS,
                         n=40), np.linspace(1.9, 1.3, 40)))
    extras.append(bud_sdf_fn(BUD_AT, BUD_AXIS))
    p, rad = strands[1]
    j = int(0.32 * len(p))
    side = -np.array([1.0, 0.6, 0]) / np.linalg.norm([1.0, 0.6])
    reach = p[j] + side * 11 + [0, 0, 12]
    arc = bezier(p[j], p[j] + side * 3 + [0, 0, 6], reach - side * 3 - [0, 0, 2], reach)
    curl, _, _ = lp.golden_curl(arc[-1], arc[-1] - arc[-2], 5.5)
    tubes.append((np.vstack([arc, curl]),
                   np.concatenate([np.linspace(2.0, 1.6, len(arc)), np.linspace(1.6, 0.9, len(curl))])))
    return [], tubes, extras


def build(voxel, stems=(), quiet=False):
    say = (lambda *a: None) if quiet else print
    _, tubes, extras = body_parts()
    ext = RF + 6
    xs = np.arange(-ext, ext, voxel, dtype=np.float32)
    ys = np.arange(-ext - 4, ext, voxel, dtype=np.float32)
    zs = np.arange(-0.6, H + 6, voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    t0 = time.time()

    def foot(x, y, z):
        rr = np.sqrt(x ** 2 + y ** 2)
        d = soft_band(rr - R0 * 0.45, np.abs(z - 1.2) - 1.2, 3.0, 1.2)
        return np.maximum(d, -z)
    add(vol, xs, ys, zs, (-R0, -R0, -1), (R0, R0, 5), foot, k=0)

    for pts, rr in tubes:
        for k0 in range(0, len(pts) - 1, 10):
            seg, rad = pts[k0:k0 + 11], rr[k0:k0 + 11]
            pad = rad.max() + 1.0
            add(vol, xs, ys, zs, seg.min(0) - pad, seg.max(0) + pad,
                lambda x, y, z, s_=seg, r_=rad: capsule_chain(x, y, z, s_, r_), k=0.6)
    say(f"  body {time.time() - t0:.0f}s")

    for f in extras:
        add(vol, xs, ys, zs, BUD_AT - 18, BUD_AT + 18, f, k=0.8)
    lo = CENTRE - [RF + 3, RF + 3, CALYX_DEPTH + 3]
    hi = CENTRE + [RF + 3, RF + 3, HT + 40]
    add(vol, xs, ys, zs, lo, hi, flower_sdf, k=0.8)
    say(f"  flower {time.time() - t0:.0f}s")

    for pts, rr in stems:
        add(vol, xs, ys, zs, pts.min(0) - 2.5, pts.max(0) + 2.5,
            lambda x, y, z, s_=pts, r_=rr: capsule_chain(x, y, z, s_, r_), k=0.8)
    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])        # flat seat
    return vol, (xs[0], ys[0], zs[0])


def plan_stems(voxel=0.3, passes=8):
    """Grow supports until nothing starts in mid-air (coarse grid): a
    question-mark curl under the flower if several spots cluster there,
    then single stems that may branch off it."""
    stems = []
    for i in range(passes):
        vol, origin = build(voxel, stems, quiet=True)
        isl = lp.find_islands(vol, voxel, origin)
        print(f"  stem pass {i + 1}: {len(isl)} island(s), {len(stems)} stem(s)")
        if not isl:
            break
        if i == 0:
            under = [p for p in isl if p[2] > CENTRE[2] - 4 and
                     np.linalg.norm(p[:2] - CENTRE[:2]) < RF + 3]
            if len(under) >= 3:
                P = np.array(under)
                low = P[np.argmin(P[:, 2])]
                stems.append(lp.support_curl(low - [0, 0, 0.3], low[:2] - CENTRE[:2],
                                             10.0, vol, voxel, origin))
                continue
        stems += lp.stems_for(isl, vol, voxel, origin)
    return stems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.12, help="mesh resolution, mm")
    ap.add_argument("--preview", action="store_true", help="skip support planning")
    ap.add_argument("--out", default=os.path.join(HERE, "hibiscus_sculpture.stl"))
    args = ap.parse_args()
    t0 = time.time()
    stems = () if args.preview else plan_stems()
    vol, origin = build(args.voxel, stems)
    mesh = decimate(mesh_volume(vol, args.voxel, origin), 0.006)
    print(f"{len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, "
          f"{time.time() - t0:.0f}s")
    mesh.export(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
