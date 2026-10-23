#!/usr/bin/env python3
"""Braided hibiscus bonsai (resin / MSLA, Saturn 4 Ultra), about 120 mm tall
in its pot. Two parts, both printed straight on the plate:

  tree   (~100 mm) a soil plug, a rope-of-braids trunk (four cords twisting
         round each other, each a three-strand plait) up to the golden
         point, where the cords split into four branches that each fork
         into two twigs. Clusters of serrated hibiscus leaves (raised
         midribs and veins), two large open flowers with solid, ruffled,
         veined petals, an eye ridge and a staminal column, and a twisted
         bud. Leaves and petals all rise from where they attach, so the
         tree needs no supports; anything that would still start in mid-air
         gets a slim stem (the same planner as the plumeria sculpture).
  pot    a tapered rectangular bonsai pot with a lip, corner feet and two
         drain holes. A 45-degree ledge inside holds the soil plug, 0.3 mm
         clearance all round.

Usage
  python3 bonsai_hibiscus.py --voxel 0.25 --preview     # quick look
  python3 bonsai_hibiscus.py                            # final STLs
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
from lattice_plumeria import PHI, BASE_CUT, add, bezier, frame_from_axis, soft_band  # noqa: E402
from plumeria_ring import decimate, smin  # noqa: E402

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))
UP = np.array([0, 0, 1.0])

# ---------------------------------------------------------------- layout
TREE_H = 101.0                       # top of the canopy above the plate
SOIL_Z = 4.5                         # soil plug thickness
PLUG = (29.0, 17.0, 6.0)             # plug half-length, half-width, corner radius
BRANCH_Z = TREE_H - (TREE_H - SOIL_Z) / PHI ** 2    # trunk : canopy = phi
STRAND_R = 1.0
CORD_R = 3.2
CORD_TURNS = 0.75
PLAIT_PITCH = 13.0
BRANCH_LEN = 25.0
TWIG_LEN = BRANCH_LEN / PHI * 1.1
FLOWER_R = 19.0                      # petal reach: flowers ~40 mm across
LEAF_L = 15.0

POT_H = 22.0
FIT = 0.3
LEDGE_Z = POT_H - 2.0 - SOIL_Z       # the plug rests here; soil sits 2 mm below the rim


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def frame(base, u, up_hint=UP):
    """Local frame at base: u along the part, w its face normal (toward
    up_hint), v across."""
    u = unit(u)
    w = up_hint - (up_hint @ u) * u
    w = unit(w) if np.linalg.norm(w) > 1e-6 else unit(np.cross(u, [1.0, 0, 0]))
    return np.asarray(base, float), u, np.cross(w, u), w


def local(fr, x, y, z):
    c, u, v, w = fr
    qx, qy, qz = x - c[0], y - c[1], z - c[2]
    return (qx * u[0] + qy * u[1] + qz * u[2], qx * v[0] + qy * v[1] + qz * v[2],
            qx * w[0] + qy * w[1] + qz * w[2])


def box(fr, lo, hi):
    c, u, v, w = fr
    corners = np.array([[a, b, d] for a in (lo[0], hi[0]) for b in (lo[1], hi[1])
                        for d in (lo[2], hi[2])])
    world = c + corners @ np.vstack([u, v, w])
    return world.min(0), world.max(0)


def rrect(x, y, hx, hy, r):
    qx, qy = np.abs(x) - (hx - r), np.abs(y) - (hy - r)
    return (np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
            + np.minimum(np.maximum(qx, qy), 0) - r)


# ----------------------------------------------------------------- leaves

_t = np.linspace(0, 1, 2001)
_LEAF_MAX = (_t ** 0.55 * (1 - _t) ** 0.85).max()


def leaf_sdf(fr, L=LEAF_L, W=5.0, T=1.0):
    """Ovate hibiscus leaf: serrated margin (teeth point to the tip), a
    gentle V-fold, a raised midrib and four pairs of raised side veins,
    on a short petiole."""
    def f(x, y, z):
        u, v, w = local(fr, x, y, z)
        t = u / L
        tc = np.clip(t, 0, 1)
        half = W * tc ** 0.55 * (1 - tc) ** 0.85 / _LEAF_MAX
        saw = (t * 9.0) % 1.0
        half = half * (1 + np.where((t > 0.1) & (t < 0.95), 0.10 * saw - 0.05, 0))
        o = np.maximum(np.abs(v) - half - 1e-3, np.maximum(-u, u - L))
        h = 0.07 * v ** 2 + 1.2 * tc ** 2                    # V-fold, tip lifts
        d = soft_band(o, np.abs(w - h) - T / 2, 3.0, T / 2)
        rib = np.maximum(np.abs(v) - 0.45, o + 0.9)
        d = smin(d, soft_band(rib, np.abs(w - h - 0.45) - 0.35, 0.45, 0.35), 0.3)
        side = np.min([np.abs(u - k * L - 0.75 * np.abs(v)) for k in (0.16, 0.33, 0.5, 0.66)],
                      axis=0) * 0.8
        vein = np.maximum(side - 0.28, o + 0.7)
        d = smin(d, soft_band(vein, np.abs(w - h - 0.4) - 0.22, 0.28, 0.22), 0.25)
        stalk = capsule_chain(u, v, w, np.array([[-3.5, 0, 0.0], [0.3, 0, 0.0]]),
                              np.array([0.75, 0.6]))
        return smin(d, stalk, 0.5)
    return f, box(fr, (-4.5, -W - 1.5, -3), (L + 1, W + 1.5, 4))


# ---------------------------------------------------------------- flowers

def flower_sdf(fr, R=FLOWER_R):
    """Open hibiscus with solid petals like the reference photo: five broad
    petals overlapping as a pinwheel, ruffled (a scalloped outline, kept in
    plane so the edge never dips, fading out at the rounded tips), a thickened rim, raised radiating veins, a ridge
    round the eye, a calyx, and an arching staminal column with anthers
    and five stigma pads."""
    rho0 = 2.5
    L = R - rho0
    ht = 0.72 * L
    col = bezier(np.array([0, 0, -1.0]), np.array([0, 0, 7.0]), np.array([1.5, 0, 13.0]),
                 np.array([4.0, 0, 18.5]), n=30)
    col_r = np.linspace(1.15, 0.8, len(col))
    beads = []
    for k in range(13):
        j = int((0.6 + 0.3 * k / 12) * (len(col) - 1))
        tan = unit(col[j + 1] - col[j - 1])
        e1 = unit(np.cross(tan, [0, 1.0, 0]))
        e2 = np.cross(tan, e1)
        a = k * GOLDEN_ANGLE
        out = np.cos(a) * e1 + np.sin(a) * e2
        beads.append((col[j], col[j] + 0.85 * col_r[j] * out + 0.9 * tan, 0.3, 0.45))   # seated in the column
    tip, tan = col[-1], unit(col[-1] - col[-3])
    e1 = unit(np.cross(tan, [0, 1.0, 0]))
    e2 = np.cross(tan, e1)
    for k in range(5):
        a = 2 * np.pi * k / 5
        beads.append((tip, tip + 1.4 * (0.9 * tan + 0.4 * (np.cos(a) * e1 + np.sin(a) * e2)),
                      0.35, 0.5))

    def f(x, y, z):
        lx, ly, lz = local(fr, x, y, z)
        rho = np.sqrt(lx ** 2 + ly ** 2) + 1e-6
        u = rho - rho0
        t = u / L
        tc = np.clip(t, 0, 1)
        s = np.clip(t, 1e-3, 1.1)
        base_h, slope = ht * s ** 0.8, 0.8 * ht / L * s ** -0.2
        ang = np.arctan2(ly, lx)
        d = np.full(lx.shape, 1e9, np.float32)
        for i in range(5):
            phi0 = 2 * np.pi * i / 5 + 0.3 * tc
            dphi = np.angle(np.exp(1j * (ang - phi0)))
            v = rho * dphi
            half = 0.82 * L * np.where(tc < 0.7, tc ** 0.6,
                                       0.807 * np.sqrt(np.clip(1 - ((tc - 0.7) / 0.3) ** 2, 0, 1))) + 1e-3
            psi = np.arctan2(v, u - 0.45 * L)
            ramp = np.clip((t - 0.5) / 0.35, 0, 1)
            o = np.maximum(np.abs(v) - half, np.maximum(-u, u - L)) + 0.45 * ramp * np.clip((0.9 - t) / 0.12, 0, 1) * np.sin(9 * psi)
            sv = np.clip(v / half, -1, 1)
            h = base_h + 0.5 * sv * tc
            dz = np.abs(lz - h) / np.sqrt(1 + slope ** 2)
            blade = soft_band(o, dz - 0.6, 4.0, 0.6)                    # 1.2 mm petal
            rim = np.maximum(o, -(o + 1.1))
            blade = smin(blade, soft_band(rim, dz - 0.8, 0.55, 0.8), 0.4)   # 1.6 mm rim
            fan = np.abs(((sv + 1) * 4.5) % 1.0 - 0.5) / 4.5 * half
            eye = np.abs(u - 0.3 * L)
            ridge = np.maximum(np.minimum(fan, eye) - 0.25,
                               np.maximum(o + 1.0, 0.12 * L - u))
            top = np.abs(lz - h - 0.62) - 0.2
            blade = smin(blade, soft_band(ridge, top, 0.25, 0.2), 0.25)
            d = smin(d, blade, 0.4)
        cup = np.sqrt(lx ** 2 + ly ** 2 + ((lz + 2.2) / 1.3) ** 2) - 3.4
        cup = np.maximum(cup, lz - 1.0)
        sc = np.clip(-lz / 8.0, 0, 1)
        cone = np.maximum(np.sqrt(lx ** 2 + ly ** 2) - (3.2 - 1.9 * sc ** 0.8),
                          np.maximum(-lz - 8.0, lz + 1.0))
        d = smin(d, smin(cup, cone, 1.0), 0.8)
        d = smin(d, capsule_chain(lx, ly, lz, col, col_r), 0.6)
        for a, b, rs, rb in beads:
            stalk = capsule_chain(lx, ly, lz, np.array([a, b]), np.array([rs, rs]))
            bead = np.sqrt((lx - b[0]) ** 2 + (ly - b[1]) ** 2 + (lz - b[2]) ** 2) - rb
            d = smin(d, smin(stalk, bead, 0.2), 0.25)
        return d
    return f, box(fr, (-R - 2, -R - 2, -9), (R + 2, R + 2, ht + 22))


def bud_sdf(base, axis, length=12.0, radius=2.8):
    fr = frame(base, axis, UP if abs(unit(axis)[2]) < 0.95 else np.array([1.0, 0, 0]))

    def f(x, y, z):
        lz, lx, ly = local(fr, x, y, z)                  # lz along the bud
        s = np.clip(lz / length, 0, 1)
        r = radius * np.sin(np.pi * np.clip(s * 0.95 + 0.05, 0, 1)) ** 0.7 * (1 - 0.35 * s)
        th = np.arctan2(ly, lx) - 4.0 * s
        r = r * (1 - 0.10 * np.abs(np.sin(2.5 * th)))
        body = np.maximum(np.sqrt(lx ** 2 + ly ** 2) - r, np.maximum(-lz, lz - length))
        tip = np.sqrt(lx ** 2 + ly ** 2 + (lz - length) ** 2) - 0.5
        sep = np.sqrt(lx ** 2 + ly ** 2 + ((lz - 2.0) / 1.6) ** 2) - radius * 0.95
        return smin(smin(body, tip, 0.8), np.maximum(sep, lz - 3.8), 0.7)
    return f, box(fr, (-2, -radius - 2, -radius - 2), (length + 2, radius + 2, radius + 2))


# ------------------------------------------------------------------ tree

def plait(cord, z, j, merge, r):
    """Three strands plaited round a cord centreline (side to side and
    over-under), closing up where merge falls to 0."""
    tan = np.gradient(cord, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    radial = cord - np.column_stack([0 * z, 0 * z, cord[:, 2]])
    radial[np.linalg.norm(radial, axis=1) < 1e-3] = [1.0, 0, 0]
    e1 = radial - (radial * tan).sum(1)[:, None] * tan
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(tan, e1)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cord, axis=0), axis=1))])
    t = 2 * np.pi * arc / PLAIT_PITCH + j * 2 * np.pi / 3
    out = []
    for k in range(3):
        ph = 2 * np.pi * k / 3
        wx = 1.6 * r * merge * np.sin(t + ph)
        wy = 1.05 * r * merge * np.sin(2 * (t + ph))
        out.append((cord + wx[:, None] * e2 + wy[:, None] * e1, r))
    return out


def tier_leaves(things, p, a0, n=3, spread=0.9, el0=32):
    for k in range(n):
        a = a0 + (k - (n - 1) / 2) * spread
        el = np.radians(el0 + 8 * (k % 3))
        ld = unit(np.cos(el) * np.array([np.cos(a), np.sin(a), 0]) + np.sin(el) * UP)
        things.append(("leaf", frame(p + 3.0 * ld, ld)))


def tree_parts():
    """Trunk strands, tiers, branches, twigs, and where flowers, bud and
    leaves go.

    Three cords, each a three-strand plait. The centre cord rises straight
    from the soil; the two side cords rise out of the soil near the pot
    edges and sweep in on an S-curve (two radii) to join it at the golden
    height. From there the three twist together, still reading as three
    into one, up to the crown, where they part into three branches. A
    middle tier sweeps out both ways as plaits into question-mark curls."""
    tubes, things = [], []
    zj = SOIL_Z + (BRANCH_Z - SOIL_Z) / PHI ** 1.5         # where the side cords join
    z2 = zj + 7.0                                          # middle tier, just above

    def helix(j, z):
        s = (z - zj) / (BRANCH_Z - zj)
        th = 2 * np.pi * (j / 3 + CORD_TURNS * s)
        r = CORD_R * (1 - 0.12 * s)
        return np.column_stack([r * np.cos(th), r * np.sin(th), z]), th

    zt = np.linspace(zj, BRANCH_Z, 220)
    tops = []
    for j in range(3):
        up, th = helix(j, zt)
        tan_j = unit(up[3] - up[0])
        if j == 1:                                         # centre cord: straight up
            foot = np.array([0, 0, SOIL_Z - 0.8])
            low = bezier(foot, foot + [0, 0, 0.5 * (zj - SOIL_Z)], up[0] - 8 * tan_j, up[0], n=60)
        else:                                              # side cords: S-curve from the pot edge
            sgn = 1 if j == 0 else -1
            foot = np.array([sgn * 23.0, 1.5 * sgn, SOIL_Z - 0.8])
            low = bezier(foot, foot + [0, 0, 16.0], up[0] - 12 * tan_j, up[0], n=110)
        cord = np.vstack([low[:-1], up])
        r = np.concatenate([np.full(len(low) - 1, STRAND_R * 0.95),
                            STRAND_R * np.linspace(0.95, 0.85, len(up))])
        tubes += plait(cord, cord[:, 2], j, np.ones(len(cord)), r)
        tops.append((cord, th[-1]))
        # a little root flare where each cord meets the soil
        out = unit([foot[0], foot[1], 0]) if np.linalg.norm(foot[:2]) > 1 else np.array([0, 1.0, 0])
        for k in (-1, 1):
            rd = unit(out + 0.9 * k * np.array([-out[1], out[0], 0]))
            root = bezier(foot + [0, 0, 2.0], foot + rd * 3 + [0, 0, 1.4],
                          foot + rd * 6 + [0, 0, 0.4], foot + rd * 8 - [0, 0, 0.6], n=14)
            tubes.append((root, np.linspace(1.4, 0.6, 14)))

    # middle tier: sweeps out both ways and rolls into question-mark curls
    p0 = np.array([0, 0, z2])
    for a in (np.radians(30), np.radians(210)):
        out = np.array([np.cos(a), np.sin(a), 0])
        tip = p0 + 17.0 * out + np.array([0, 0, 11.0])
        arm = bezier(p0, p0 + 5 * out + [0, 0, 1], tip - 5 * out - [0, 0, 3], tip, n=70)
        merge = np.clip((len(arm) - 1 - np.arange(len(arm))) / 10.0, 0, 1)   # closes up at the curl
        tubes += plait(arm, arm[:, 2], 5 + int(a > 1), merge, np.linspace(0.85, 0.7, len(arm)))
        tubes.append((arm[-12:], np.linspace(0.6, 1.2, 12)))
        curl, _, _ = lp.golden_curl(tip, tip - arm[-2], 4.5)
        tubes.append((curl, np.linspace(1.2, 0.8, len(curl))))
        tier_leaves(things, arm[int(0.55 * len(arm))], a, n=3, spread=0.9)
        tier_leaves(things, arm[int(0.85 * len(arm))], a + 0.5, n=2, spread=1.8, el0=40)

    # crown: the three cords part into three branches
    cords = tops
    roles = {(0, 0): "flower", (1, 1): "flower", (2, 0): "bud"}   # flowers face away from each other
    twig_ends = []
    for j, (cord, th) in enumerate(cords):
        p0 = cord[-1]
        out = np.array([np.cos(th + 0.4), np.sin(th + 0.4), 0])
        d = unit(np.cos(np.radians(40)) * out + np.sin(np.radians(40)) * UP)
        p3 = p0 + BRANCH_LEN * d
        br = bezier(p0, p0 + 6 * UP, p3 - 5 * d, p3, n=30)
        tubes.append((br, np.linspace(2.1, 1.35, 30)))
        fork = br[int(len(br) / PHI)]
        for side, sgn in enumerate((-1, 1)):
            a = th + 0.4 + sgn * 0.6
            el = np.radians(38 + 10 * side)
            if roles.get((j, side)) == "flower":
                el = np.radians(58)
            td = unit(np.cos(el) * np.array([np.cos(a), np.sin(a), 0]) + np.sin(el) * UP)
            tip = fork + TWIG_LEN * td
            tw = bezier(fork, fork + 4 * unit(d + UP), tip - 4 * td, tip, n=24)
            tubes.append((tw, np.linspace(1.35, 0.95, 24)))
            twig_ends.append((tip, td, roles.get((j, side), "leaves"), j))
            for k in range(2):                   # two leaves along each twig
                q = tw[int((0.45 + 0.25 * k) * (len(tw) - 1))]
                b = a + (1.4 if k else -1.4)
                ld = unit(np.cos(np.radians(36)) * np.array([np.cos(b), np.sin(b), 0])
                          + np.sin(np.radians(36)) * UP)
                things.append(("leaf", frame(q + 3.0 * ld, ld)))
        # two leaves along each branch
        for k, f_ in enumerate((0.35, 0.8)):
            p = br[int(f_ * (len(br) - 1))]
            a = th + 0.4 + (1.3 if k else -1.3)
            ld = unit(0.55 * np.array([np.cos(a), np.sin(a), 0]) + 0.45 * d + 0.5 * UP)
            things.append(("leaf", frame(p + 3.0 * ld, ld)))

    for tip, td, role, j in twig_ends:
        if role == "flower":
            axis = unit(0.14 * np.array([td[0], td[1], 0]) + UP)
            things.append(("flower", frame(tip + 2.0 * axis, [1.0, 0, 0] - axis[0] * axis, axis)))
            # a leaf or two tucked under each flower
            for k in range(2):
                a = np.arctan2(td[1], td[0]) + (k - 0.5) * 2.2
                ld = unit(0.75 * np.array([np.cos(a), np.sin(a), 0]) + 0.6 * UP)
                things.append(("leaf", frame(tip - 4.0 * td + 3.0 * ld, ld)))
        elif role == "bud":
            things.append(("bud", (tip, unit(td + 0.6 * UP))))
            ld = unit(np.array([-td[1], td[0], 0]) + 0.7 * UP)
            things.append(("leaf", frame(tip - 3.0 * td + 2.4 * ld, ld)))
        else:
            # three short sprigs fan out from the twig tip, each ending in a
            # rosette of leaves, so the crown fills out into a dome
            a0 = np.arctan2(td[1], td[0])
            for m, (da, el_s) in enumerate(((-0.75, 35), (0.75, 35), (0.0, 62))):
                sd = unit(np.cos(np.radians(el_s)) * np.array([np.cos(a0 + da), np.sin(a0 + da), 0])
                          + np.sin(np.radians(el_s)) * UP)
                sp = bezier(tip, tip + 2.5 * td, tip + 7.5 * sd - 2.0 * sd, tip + 7.5 * sd, n=14)
                tubes.append((sp, np.linspace(0.95, 0.8, 14)))
                end = sp[-1]
                for k in range(3):
                    a = a0 + da + (k - 1) * 0.9 + 0.25 * np.sin(j + m + k)
                    el = np.radians(32 + 8 * ((k + m + j) % 3))
                    ld = unit(np.cos(el) * np.array([np.cos(a), np.sin(a), 0]) + np.sin(el) * UP)
                    things.append(("leaf", frame(end + 3.0 * ld, ld)))
    return tubes, [t for t in things if not clips_flower(t, things)]


def clips_flower(thing, things):
    """True for a leaf whose blade would pass through a flower's bowl."""
    if thing[0] != "leaf":
        return False
    base, u, _, _ = thing[1]
    pts = base + np.linspace(-0.2, 1.0, 10)[:, None] * LEAF_L * u
    for kind, fr in things:
        if kind != "flower":
            continue
        q = pts - fr[0]
        h = q @ fr[3]
        r = np.linalg.norm(q - h[:, None] * fr[3], axis=1)
        if np.any((r < FLOWER_R + 1.5) & (h > -4) & (h < 0.72 * FLOWER_R + 3)):
            return True
    return False


def build_tree(voxel, stems=(), quiet=False):
    say = (lambda *a: None) if quiet else print
    t0 = time.time()
    tubes, things = tree_parts()
    ext = 44.0
    xs = np.arange(-ext, ext, voxel, dtype=np.float32)
    ys = np.arange(-ext, ext, voxel, dtype=np.float32)
    zs = np.arange(-0.6, TREE_H + 8, voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)

    def plug(x, y, z):
        bumps = 0.25 * np.sin(1.7 * x) * np.sin(1.9 * y) + 0.15 * np.sin(3.1 * x + 1) * np.sin(2.7 * y)
        return np.maximum(rrect(x, y, *PLUG), np.maximum(-z, z - SOIL_Z - bumps))
    add(vol, xs, ys, zs, (-PLUG[0] - 1, -PLUG[1] - 1, -1), (PLUG[0] + 1, PLUG[1] + 1, SOIL_Z + 1),
        plug, k=0)

    for pts, rr in tubes:
        rr = np.broadcast_to(rr, (len(pts),))
        for k0 in range(0, len(pts) - 1, 10):
            seg, rad = pts[k0:k0 + 11], rr[k0:k0 + 11]
            pad = rad.max() + 1.0
            add(vol, xs, ys, zs, seg.min(0) - pad, seg.max(0) + pad,
                lambda x, y, z, s_=seg, r_=rad: capsule_chain(x, y, z, s_, r_), k=0.6)
    say(f"  trunk and branches {time.time() - t0:.0f}s")
    for kind, spec in things:
        if kind == "leaf":
            f, (lo, hi) = leaf_sdf(spec)
        elif kind == "bud":
            f, (lo, hi) = bud_sdf(*spec)
        else:
            f, (lo, hi) = flower_sdf(spec)
        add(vol, xs, ys, zs, lo, hi, f, k=0.5, chunk=8)
    say(f"  leaves, flowers, bud {time.time() - t0:.0f}s")
    for pts, rr in stems:
        add(vol, xs, ys, zs, pts.min(0) - 2.5, pts.max(0) + 2.5,
            lambda x, y, z, s_=pts, r_=rr: capsule_chain(x, y, z, s_, r_), k=0.8)
    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])
    return vol, (xs[0], ys[0], zs[0])


def clear_stems(islands, vol, voxel, origin):
    """Like the sculpture's stem planner, but a stem may only take a path
    that stays clear of everything between its foot and its tip, so no
    stem pokes through a petal or leaf on the way up."""
    from scipy import ndimage as ndi
    S = vol < 0
    pts = np.argwhere(S & ~ndi.binary_erosion(S)) * voxel + np.asarray(origin)
    org = np.asarray(origin)

    def clear(path):
        mid = path[int(0.12 * len(path)):int(0.82 * len(path))]
        idx = np.round((mid - org) / voxel).astype(int)
        idx = np.clip(idx, 0, np.array(vol.shape) - 1)
        return not np.any(vol[idx[:, 0], idx[:, 1], idx[:, 2]] < -0.15)

    stems = []
    for p in sorted(islands, key=lambda q: -q[2]):
        top = p + [0, 0, 0.5]
        drop = top[2] - pts[:, 2]
        horiz = np.linalg.norm(pts[:, :2] - top[:2], axis=1)
        ok = np.nonzero((drop > 2.0) & (horiz < 0.7 * drop))[0]
        order = ok[np.argsort(np.sqrt(drop[ok] ** 2 + horiz[ok] ** 2))]
        for i in order[:400:4]:
            path = lp.stem_path(top, pts[i] - [0, 0, 0.6])
            if clear(path):
                stems.append((path, np.linspace(*lp.STEM_R, len(path))))
                pts = np.vstack([pts, path[path[:, 2] < top[2] - 4.0] + [0, 0, 0.6]])
                break
        else:
            print(f"  no clear stem path for island at {np.round(p, 1)}")
    return stems


def plan_stems(voxel=0.3, passes=8):
    """Slim stems up to anything that would start in mid-air (coarse grid)."""
    stems = []
    for i in range(passes):
        vol, origin = build_tree(voxel, stems, quiet=True)
        isl = lp.find_islands(vol, voxel, origin)
        print(f"  stem pass {i + 1}: {len(isl)} island(s), {len(stems)} stem(s)")
        if not isl:
            break
        stems += clear_stems(isl, vol, voxel, origin)
    return stems


# ------------------------------------------------------------------- pot

def build_pot(voxel):
    xs = np.arange(-40, 40, voxel, dtype=np.float32)
    ys = np.arange(-27, 27, voxel, dtype=np.float32)
    zs = np.arange(-0.6, POT_H + 1.5, voxel, dtype=np.float32)
    vol = np.full((len(xs), len(ys), len(zs)), 3.0, np.float32)
    hx0, hy0 = PLUG[0] + FIT + 4.2, PLUG[1] + FIT + 4.2

    def pot(x, y, z):
        k = z / POT_H
        lip = 1.2 * np.clip((z - (POT_H - 3.7)) / 1.2, 0, 1)        # 45-degree underside
        hx, hy = hx0 - 1.0 + 1.0 * k + lip, hy0 - 1.0 + 1.0 * k + lip
        feet = (np.abs(x) > hx - 9) & (np.abs(y) > hy - 7)
        inset = np.where(feet, 0.0, 1.0 * np.clip(3.5 - z, 0, 1))   # recess between feet
        outer = np.maximum(rrect(x, y, hx - inset, hy - inset, 6.0), np.maximum(-z, z - POT_H))
        # cavity: plug seat at the top, a 45-degree ledge, then a hollow
        seat = np.maximum(rrect(x, y, PLUG[0] + FIT, PLUG[1] + FIT, PLUG[2] + FIT), LEDGE_Z - z)
        shrink = 2.0 * np.clip((LEDGE_Z - z) / 2.0, 0, 1)
        hollow = np.maximum(rrect(x, y, PLUG[0] + FIT - shrink, PLUG[1] + FIT - shrink, 4.5),
                            np.maximum(2.5 - z, z - LEDGE_Z))
        drains = np.minimum(np.hypot(x - 12, y), np.hypot(x + 12, y)) - 2.5
        cut = np.minimum(np.minimum(seat, hollow), np.maximum(drains, z - 3.0))
        return np.maximum(outer, -cut)
    add(vol, xs, ys, zs, (-40, -27, -1), (40, 27, POT_H + 1), pot, k=0)
    vol = np.maximum(vol, (BASE_CUT - zs)[None, None, :])
    return vol, (xs[0], ys[0], zs[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.12, help="mesh resolution, mm")
    ap.add_argument("--preview", action="store_true", help="skip support planning")
    ap.add_argument("--part", choices=["tree", "pot", "both"], default="both")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()
    for part in (["tree", "pot"] if args.part == "both" else [args.part]):
        t0 = time.time()
        if part == "tree":
            vol, origin = build_tree(args.voxel, () if args.preview else plan_stems())
        else:
            vol, origin = build_pot(args.voxel)
        mesh = decimate(mesh_volume(vol, args.voxel, origin), 0.006)
        print(f"{part}: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
              f"volume={mesh.volume / 1000:.2f} mL, size={np.round(mesh.extents, 1)} mm, "
              f"{time.time() - t0:.0f}s")
        out = os.path.join(args.outdir, f"hibiscus_bonsai_{part}.stl")
        mesh.export(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
