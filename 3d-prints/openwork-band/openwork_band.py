#!/usr/bin/env python3
"""Hawaiian openwork (pierced) plumeria band generator (resin / MSLA).

A tapered band, wide across the front and narrowing to a plain back.
Between two solid edge rails the front is pierced right through, and
plumerias, sweeping stems and curling waves stand as solid metal against
the voids: an interplay of positive and negative space. The flowers carry
engraved rays on each petal and a domed centre; the sweeps carry a centre
line. ALOHA is engraved inside the plain back.

The artwork is drawn as a 2D mask on the unrolled band (s = arc length,
y = across), turned into a 2D signed distance, wrapped onto the ring as a
3D SDF, meshed with marching cubes and written as a binary STL.

Usage
  python3 openwork_band.py --size 7
  python3 openwork_band.py --size 7 --text ""
"""

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import (binary_dilation, binary_opening, distance_transform_edt,
                           gaussian_filter, label)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plumeria-ring"))
sys.path.insert(0, os.path.join(HERE, "..", "heirloom-band"))
from plumeria_ring import (build_mesh, decimate, petal_shape,  # noqa: E402
                           text_sdf_2d, us_size_to_inner_diameter)
from heirloom_band import DOME, GROOVE, RES, Motif, smax  # noqa: E402

WALL = 1.45                       # band thickness
HW_FRONT, HW_BACK = 3.5, 2.75    # half width: 7 mm at the centre flower
TAPER = 14.3                      # ... easing to 6 mm where the openwork ends
RAIL = 0.6                        # slim edge rails
OW_END = 15.0                     # 30 mm decorative section, |s| < OW_END
VINE = 0.72                       # vine thickness
TEXT_HEIGHT, TEXT_DEPTH = 1.5, 0.25
ENGRAVE = 0.12                    # extra depth of engraved lines


def half_width(s):
    return HW_BACK + (HW_FRONT - HW_BACK) * np.exp(-(s / TAPER) ** 2)


def catmull_rom(pts, n=24):
    """Smooth curve through control points (endpoints repeated)."""
    p = np.asarray(pts, float)
    p = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(p) - 2):
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        a, b, c, d = p[i - 1], p[i], p[i + 1], p[i + 2]
        out.append(0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t ** 2
                          + (-a + 3 * b - 3 * c + d) * t ** 3))
    out.append(p[-2][None])
    return np.vstack(out)


class Openwork(Motif):
    """Motif on the full circumference, drawn in signed arc length s."""

    def __init__(self, circ):
        super().__init__(circ, HW_FRONT)
        self.circ = circ
        self.Ss = (self.S + circ / 2) % circ - circ / 2    # signed arc length
        self.bump = np.zeros(self.S.shape)                  # extra relief, mm

    def stroke(self, pts, w0, w1, curl=None):
        """Tapered sweep through `pts`, optionally ending in a curling wave.

        curl = (radius, turns, sense): a spiral continuing from the end of
        the sweep, turning left (+1) or right (-1).
        """
        path = catmull_rom(pts)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(seg)])
        widths = w0 + (w1 - w0) * arc / arc[-1]
        if curl is not None:
            r0, turns, sense = curl
            tan = path[-1] - path[-2]
            tan /= np.linalg.norm(tan)
            nrm = sense * np.array([-tan[1], tan[0]])
            centre = path[-1] + r0 * nrm
            a0 = np.arctan2(*(path[-1] - centre)[::-1])
            th = np.linspace(0, 2 * np.pi * turns, int(60 * turns))[1:]
            rr = r0 * (1 - 0.75 * th / th[-1])
            spiral = centre + np.column_stack([np.cos(a0 + sense * th),
                                               np.sin(a0 + sense * th)]) * rr[:, None]
            path = np.vstack([path, spiral])
            widths = np.concatenate([widths, w1 * (1 - 0.45 * th / th[-1])])

        pad = widths.max() + 0.1
        lo, hi = path.min(0) - pad, path.max(0) + pad
        win = (self.Ss > lo[0]) & (self.Ss < hi[0]) & (self.Y > lo[1]) & (self.Y < hi[1])
        ix = np.nonzero(win)
        ps, py = self.Ss[ix], self.Y[ix]
        best = np.full(ps.shape, 1e9)
        wbest = np.zeros(ps.shape)
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            ab = b - a
            L2 = max(ab @ ab, 1e-12)
            t = np.clip(((ps - a[0]) * ab[0] + (py - a[1]) * ab[1]) / L2, 0, 1)
            d = np.hypot(ps - a[0] - t * ab[0], py - a[1] - t * ab[1])
            closer = d < best
            best = np.where(closer, d, best)
            wbest = np.where(closer, widths[i] + t * (widths[i + 1] - widths[i]), wbest)
        m = np.zeros(self.S.shape, bool)
        g = np.zeros(self.S.shape, bool)
        m[ix] = best < wbest / 2
        g[ix] = (best < GROOVE * 0.7) & (wbest > 0.62)
        vis = self.paint(m)
        self.groove |= g & vis

    def flower(self, s0, y0, radius, rot, bead=False):
        """Plumeria with engraved rays on each petal and a domed centre
        (a raised bead centre if `bead`)."""
        k = radius / 7.5
        x, y = (self.Ss - s0) / k, (self.Y - y0) / k
        petals = []
        for i in range(5):
            th = rot + 2 * np.pi * i / 5
            dx, dy = np.cos(th), np.sin(th)
            px, py = x + 1.0 * dy, y - 1.0 * dx
            u, v = px * dx + py * dy, -px * dy + py * dx
            t = np.clip(u / 7.5, 0, 1)
            w = 3.3 * petal_shape(t) * np.where(v > 0, 1.1, 0.9)
            m = (np.abs(v) < w) & (u > 0) & (u < 7.5)
            petals.append(m)
            for a in np.linspace(-0.32, 0.32, 5):            # fan of rays
                self.groove |= m & (np.abs(v - np.tan(a) * u) < GROOVE / k * 0.7) \
                    & (t > 0.22) & (t < 0.62 - 0.25 * abs(a))
        for m in petals:
            self.paint(m)
        rr = np.hypot(x, y)
        self.paint(rr < 1.6)
        self.groove |= np.abs(rr - 1.6) < GROOVE / k
        if bead:
            self.bump = np.maximum(self.bump, 0.35 * radius / 2.0
                                   * np.sqrt(np.clip(1 - (rr / 1.5) ** 2, 0, 1)))


def compose(circ):
    """Front section laid out after the flat-layout sheet (30 x 6 mm)."""
    m = Openwork(circ)
    hw = half_width(m.Ss)
    band = np.abs(m.Y) < hw
    field = (np.abs(m.Y) < hw - RAIL) & (np.abs(m.Ss) < OW_END)
    m.paint(band & (np.abs(m.Ss) >= OW_END))                       # plain shank
    m.paint(band & (np.abs(m.Y) >= hw - RAIL))                      # rails

    for sg in (1, -1):                       # mirror-image sides
        def P(*pts):
            return [(sg * s, y) for s, y in pts]
        # long vine arcing out along the top rail and dropping to the curls
        m.stroke(P((1.6, 2.1), (4.2, 2.75), (6.8, 2.6), (8.1, 1.8), (8.3, 0.8)),
                 VINE, VINE * 0.85, curl=(0.62, 1.05, sg))
        # S-vine: out of the flower, dipping low, rolling up into a big curl
        m.stroke(P((2.4, -0.4), (4.2, -1.3), (5.9, -1.0), (7.0, 0.2), (6.6, 1.25)),
                 VINE, VINE * 0.9, curl=(0.72, 1.35, sg))
        # lower vine along the bottom rail, ending in the second small curl
        m.stroke(P((1.8, -2.1), (5.0, -2.6), (7.4, -2.4), (8.7, -1.8), (9.0, -1.1)),
                 VINE, VINE * 0.8, curl=(0.55, 1.05, -sg))
        # veined leaf resting on the lower vine
        m.leaf(sg * 4.2, -2.1, 0.45 if sg > 0 else np.pi - 0.45, 2.5, 0.68)
        # crossing vines just before the shank
        m.stroke(P((13.0, 2.6), (13.9, 0.8), (14.4, -1.2), (15.1, -2.6)), VINE * 0.85,
                 VINE * 0.85)
        m.stroke(P((13.0, -2.6), (13.9, -0.8), (14.4, 1.2), (15.1, 2.6)), VINE * 0.85,
                 VINE * 0.85)
        # stems holding the side flower to the upper and lower vines
        m.stroke(P((7.6, 2.55), (9.6, 2.0), (11.0, 1.0)), VINE * 0.8, VINE * 0.8)
        m.stroke(P((7.9, -2.4), (9.8, -1.9), (11.0, -0.9)), VINE * 0.8, VINE * 0.8)
        m.flower(sg * 11.75, 0.1, 1.6, 0.3 if sg > 0 else 0.3 + np.pi / 5, bead=True)
    m.flower(0.0, 0.0, 2.9, np.pi / 2, bead=True)
    m.solid = m.raised & band
    return m


def check(m):
    """Fill slivers of void too tight to print, drop floating islands and
    report any strut thinner than 0.7 mm."""
    inside = np.abs(m.Y) < half_width(m.Ss)
    void = inside & ~m.solid
    m.solid = m.solid | (void & ~binary_opening(void, iterations=int(0.25 / RES)))
    lab, n = label(m.solid)
    sizes = np.bincount(lab.ravel())[1:]
    keep = lab == (np.argmax(sizes) + 1)
    islands = n - 1
    thin = keep & ~binary_opening(keep, iterations=int(0.35 / RES))
    print(f"openwork check: {islands} floating islands removed, "
          f"{thin.sum() * RES * RES:.2f} mm2 of strut under 0.7 mm")
    m.solid = keep
    return m


class OpenworkBand:
    def __init__(self, size, text="ALOHA"):
        self.ri = us_size_to_inner_diameter(size) / 2
        self.ro = self.ri + WALL
        self.rout = self.ro
        circ = 2 * np.pi * self.ro
        m = check(compose(circ))
        self.m = m
        self.y0 = m.y[0]
        self.n = m.S.shape[0]

        # 2D signed distance to the solid (periodic in s)
        solid = m.solid
        tiled = np.concatenate([solid] * 3)
        sd = (distance_transform_edt(~tiled) - distance_transform_edt(tiled)) * RES
        self.sd = sd[self.n:2 * self.n]

        # relief on the solid: bevelled pieces split by engraved lines
        edges = np.zeros(solid.shape, bool)
        edges[1:] |= m.label[1:] != m.label[:-1]
        edges[:, 1:] |= m.label[:, 1:] != m.label[:, :-1]
        edges = binary_dilation(edges & solid, iterations=max(1, int(GROOVE / RES) - 1))
        cut = (m.groove | edges) & solid
        body = solid & ~cut
        edt = distance_transform_edt(body) * RES
        h = np.where(body, DOME * np.clip(edt / 0.35, 0, 1) ** 0.7 - DOME, -DOME - ENGRAVE)
        self.h = gaussian_filter(np.where(solid, h + m.bump, 0.0), 0.8)
        self.text = text_sdf_2d(text, TEXT_HEIGHT)

    def sample(self, arr, s, y):
        fs = (s % self.m.circ) / RES
        fy = np.clip((y - self.y0) / RES, 0, arr.shape[1] - 1.001)
        i0 = fs.astype(int) % self.n
        i1 = (i0 + 1) % self.n
        j0 = fy.astype(int)
        a, b = fs - np.floor(fs), fy - j0
        return (arr[i0, j0] * (1 - a) * (1 - b) + arr[i1, j0] * a * (1 - b)
                + arr[i0, j0 + 1] * (1 - a) * b + arr[i1, j0 + 1] * a * b)

    def sdf(self, x, y, z):
        r = np.sqrt(x * x + z * z)
        phi = np.arctan2(x, z) % (2 * np.pi)
        s = phi * self.ro
        d2 = self.sample(self.sd, s, y)
        outer = r - (self.ro + self.sample(self.h, s, y))
        d = smax(outer, d2, 0.3)                          # softened top edges
        d = np.maximum(d, smax(self.ri - r, d2, 0.35))    # comfort-fit inside
        if self.text is not None:                         # inside, at the back
            si = -(phi - np.pi) * self.ri
            dt = np.maximum(self.text(si, y), r - (self.ri + TEXT_DEPTH))
            d = np.maximum(d, -dt)
        return np.maximum(d, self.ri - r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=7.0, help="US ring size")
    ap.add_argument("--voxel", type=float, default=0.04, help="mesh resolution, mm")
    ap.add_argument("--tolerance", type=float, default=0.004,
                    help="max simplification deviation, mm")
    ap.add_argument("--text", default="ALOHA", help='inner engraving ("" for none)')
    ap.add_argument("--metal", action="store_true",
                    help="for casting or electroplating: thicker wires, deeper engravings")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.metal:
        global VINE, RAIL, GROOVE, ENGRAVE, TEXT_DEPTH
        VINE, RAIL, GROOVE, ENGRAVE, TEXT_DEPTH = 1.0, 0.85, 0.09, 0.28, 0.40

    band = OpenworkBand(args.size, args.text)
    print(f"US size {args.size}: inner diameter {2*band.ri:.2f} mm")
    mesh = build_mesh(band, args.voxel, y_half=HW_FRONT + 0.4,
                      z_top=band.ro + 0.8, pad=0.4)
    print(f"marching cubes: {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
    mesh = decimate(mesh, args.tolerance)
    print(f"final: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume/1000:.3f} mL, bounds={np.round(mesh.extents, 2)} mm")
    out = args.out or f"openwork_band_US{args.size:g}{'_metal' if args.metal else ''}.stl"
    mesh.export(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
