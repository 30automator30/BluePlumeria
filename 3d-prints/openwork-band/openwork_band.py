#!/usr/bin/env python3
"""Hawaiian openwork (pierced) plumeria band generator (resin / MSLA).

A tapered band (7.5 mm at the front, ~6 mm at the sides, ~5 mm plain
shank at the back) framed by slim rounded rails. Across the front the band
is pierced right through: a sculpted centre plumeria floats free of the
rails, held only by rounded vines that loop out from behind it into
curling waves, with veined leaves in the loops and a smaller plumeria on
each side before the band closes into a solid shank. Optional prong
setting for a 2-2.5 mm round stone in the centre flower. ALOHA is
engraved inside the back.

The artwork is drawn as a 2D mask plus a sculpted height map on the
unrolled band (s = arc length, y = across), wrapped onto the ring as a
3D SDF, meshed with marching cubes and written as a binary STL.

Usage
  python3 openwork_band.py --size 7
  python3 openwork_band.py --size 7 --stone 2.5     # with stone setting
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
from plumeria_ring import (build_mesh, decimate, petal_shape, smin,  # noqa: E402
                           text_sdf_2d, us_size_to_inner_diameter)
from heirloom_band import RES, smax  # noqa: E402

WALL = 1.3                        # band thickness at the rails
HW_FRONT, HW_BACK = 3.75, 2.5     # half width at the front / back
TAPER = 15.6                      # arc length over which the band narrows
RAIL_FRONT, RAIL_BACK = 0.55, 0.75  # rail width, slimmest at the front
OW_END = 17.8                     # openwork spans |s| < OW_END
VINE = 0.62                       # vine thickness
VEIN = 0.035                      # half width of fine engraved lines
TEXT_HEIGHT, TEXT_DEPTH = 1.5, 0.25


def half_width(s):
    return HW_BACK + (HW_FRONT - HW_BACK) * np.exp(-(s / TAPER) ** 2)


def rail_width(s):
    f = np.exp(-(s / TAPER) ** 2)
    return RAIL_BACK + (RAIL_FRONT - RAIL_BACK) * f


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


class Canvas:
    """Unrolled band: solid mask, sculpted height (mm, 0 = rail top)."""

    def __init__(self, circ):
        self.circ = circ
        self.s = np.arange(0, circ, RES)
        self.y = np.arange(-HW_FRONT - 0.3, HW_FRONT + 0.3, RES)
        S, self.Y = np.meshgrid(self.s, self.y, indexing="ij")
        self.Ss = (S + circ / 2) % circ - circ / 2        # signed arc length
        self.solid = np.zeros(S.shape, bool)
        self.h = np.zeros(S.shape)
        self.cut = np.zeros(S.shape)                       # engraved depth
        self.label = np.zeros(S.shape, np.int16)
        self.n = 1

    def paint(self, mask, height, behind=False):
        if behind:
            mask = mask & ~self.solid
        self.solid |= mask
        self.h = np.where(mask, height, self.h)
        self.cut = np.where(mask, 0.0, self.cut)
        self.label[mask] = self.n
        self.n += 1
        return mask

    def engrave(self, mask, depth):
        self.cut = np.where(mask, np.maximum(self.cut, depth), self.cut)

    # -------------------------------------------------------------- strokes
    def vine(self, pts, w0, w1, curl=None, behind=False):
        """Rounded vine through `pts`, optionally ending in a spiral curl.

        curl = (radius, turns, sense): spiral turning left (+1) / right (-1).
        """
        path = catmull_rom(pts)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(seg)])
        widths = w0 + (w1 - w0) * arc / arc[-1]
        if curl is not None:
            r0, turns, sense = curl
            tan = path[-1] - path[-2]
            tan /= np.linalg.norm(tan)
            centre = path[-1] + r0 * sense * np.array([-tan[1], tan[0]])
            a0 = np.arctan2(*(path[-1] - centre)[::-1])
            th = np.linspace(0, 2 * np.pi * turns, int(70 * turns))[1:]
            rr = r0 * (1 - 0.72 * th / th[-1])
            spiral = centre + np.column_stack([np.cos(a0 + sense * th),
                                               np.sin(a0 + sense * th)]) * rr[:, None]
            path = np.vstack([path, spiral])
            widths = np.concatenate([widths, w1 * (1 - 0.35 * th / th[-1])])
        d, w = self._polyline_dist(path, widths)
        m = d < w / 2
        prof = np.sqrt(np.clip(1 - (2 * d / np.maximum(w, 1e-6)) ** 2, 0, 1))
        return self.paint(m, -0.32 + 0.42 * prof, behind)

    def _polyline_dist(self, path, widths):
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
            dd = np.hypot(ps - a[0] - t * ab[0], py - a[1] - t * ab[1])
            closer = dd < best
            best = np.where(closer, dd, best)
            wbest = np.where(closer, widths[i] + t * (widths[i + 1] - widths[i]), wbest)
        d = np.full(self.Ss.shape, 1e9)
        w = np.zeros(self.Ss.shape)
        d[ix], w[ix] = best, wbest
        return d, w

    def leaf(self, s0, y0, ang, length, half_w, behind=False):
        """Veined leaf with a rounded, sculpted surface."""
        ca, sa = np.cos(ang), np.sin(ang)
        u = (self.Ss - s0) * ca + (self.Y - y0) * sa
        v = -(self.Ss - s0) * sa + (self.Y - y0) * ca
        t = np.clip(u / length, 0, 1)
        w = half_w * np.sin(np.pi * t ** 0.8) ** 0.8 + 1e-6
        m = (u > 0) & (u < length) & (np.abs(v) < w)
        vn = np.clip(np.abs(v) / w, 0, 1)
        vis = self.paint(m, -0.3 + 0.42 * np.sqrt(1 - vn ** 2) * (0.6 + 0.4 * np.sin(np.pi * t)),
                         behind)
        self.engrave(vis & (np.abs(v) < VEIN * 1.2) & (t > 0.08) & (t < 0.9), 0.1)
        side = np.abs((((u - 1.3 * np.abs(v)) / 0.45) % 1.0) - 0.5) < 0.09
        self.engrave(vis & side & (vn > 0.15) & (vn < 0.8) & (t > 0.15) & (t < 0.85), 0.06)

    def flower(self, s0, y0, radius, rot, lift):
        """Sculpted plumeria: domed petals with fine veins, bead centre."""
        k = radius / 7.5
        x, y = (self.Ss - s0) / k, (self.Y - y0) / k
        rr = np.hypot(x, y)
        for i in range(5):
            th = rot + 2 * np.pi * i / 5
            dx, dy = np.cos(th), np.sin(th)
            px, py = x + 1.0 * dy, y - 1.0 * dx
            u, v = px * dx + py * dy, -px * dy + py * dx
            t = np.clip(u / 7.5, 0, 1)
            w = 3.3 * petal_shape(t) * np.where(v > 0, 1.1, 0.9) + 1e-6
            m = (np.abs(v) < w) & (u > 0) & (u < 7.5)
            vn = np.clip(v / w, -1, 1)
            # petal rises from the centre, domes, and its overlapping edge
            # lifts slightly; the far edge tucks under the next petal
            hp = (lift + 0.18 * (1 - t) + 0.32 * np.sqrt(np.clip(1 - vn ** 2, 0, 1))
                  * np.sin(np.pi * np.clip(t * 1.1, 0, 1)) ** 0.5 + 0.08 * vn)
            vis = self.paint(m, hp)
            for a in np.linspace(-0.3, 0.3, 7):                # fine veins
                self.engrave(vis & (np.abs(v - np.tan(a) * u) < VEIN / k)
                             & (t > 0.24) & (t < 0.72 - 0.3 * abs(a)), 0.07)
        rc = 1.45
        bead = rr < rc
        self.paint(bead, lift + 0.32 + 0.3 * np.sqrt(np.clip(1 - (rr / rc) ** 2, 0, 1)))


def compose(circ):
    c = Canvas(circ)
    Ss, Y = c.Ss, c.Y
    hw, rw = half_width(Ss), rail_width(Ss)
    band = np.abs(Y) < hw

    # solid shank around the back, closing the openwork with a soft arch
    end = OW_END + 0.7 * (Y / hw) ** 2
    shank = band & (np.abs(Ss) >= end)
    c.paint(shank, np.full(Ss.shape, -0.08))
    # slim rounded rails, running right round the ring as ridges
    dr = np.abs(np.abs(Y) - (hw - rw / 2))
    rail = band & (dr < rw / 2)
    c.paint(rail, -0.3 + 0.32 * np.sqrt(np.clip(1 - (2 * dr / rw) ** 2, 0, 1)))

    for sg in (1, -1):                       # mirror-image sides
        def P(*pts):
            return [(sg * s, y) for s, y in pts]

        def leaf(s, y, ang, ln, w):
            a = ang if sg > 0 else np.pi - ang
            c.leaf(sg * s, y, a, ln, w, behind=True)

        # upper vine: from behind the flower, kissing the top rail, then
        # arching down and rolling up into a wave curl
        c.vine(P((1.2, 1.3), (3.2, 2.45), (5.8, 2.85), (8.2, 2.2), (9.2, 0.9)),
               VINE, VINE * 0.9, curl=(0.72, 1.3, sg), behind=True)
        # lower vine: loops along the bottom rail and curls up beside the
        # side flower
        c.vine(P((1.4, -1.3), (3.6, -2.55), (6.6, -2.95), (9.8, -2.3),
                 (11.6, -0.6)), VINE, VINE * 0.9, curl=(0.62, 1.25, -sg), behind=True)
        # crossing vine linking the two loops
        c.vine(P((9.6, 2.6), (10.6, 1.4), (10.4, -0.4), (9.6, -2.2)),
               VINE * 0.85, VINE * 0.85, behind=True)
        # leaves resting in the loops
        leaf(3.9, -2.25, 0.55, 2.4, 0.62)
        leaf(6.0, 2.55, -0.65, 2.0, 0.55)
        # side plumeria held by short stems to both rails
        c.vine(P((13.0, -2.6), (13.6, -1.6), (14.4, -1.0)), VINE * 0.8, VINE * 0.8,
               behind=True)
        c.vine(P((15.3, 0.9), (16.2, 1.9), (16.6, 2.6)), VINE * 0.8, VINE * 0.8,
               behind=True)
        c.vine(P((12.6, 2.7), (12.9, 1.9)), VINE * 0.8, VINE * 0.7,
               curl=(0.45, 1.1, -sg), behind=True)
        c.flower(sg * 14.6, 0.0, 1.8, 0.4 if sg > 0 else 0.4 + np.pi / 5, -0.1)

    c.flower(0.0, 0.0, 2.6, np.pi / 2, 0.05)
    c.solid &= band
    return c


def check(c):
    """Fill slivers of void too tight to print; refuse floating islands."""
    band = np.abs(c.Y) < half_width(c.Ss)
    void = band & ~c.solid
    sliver = void & ~binary_opening(void, iterations=int(0.22 / RES))
    c.solid |= sliver
    c.h = np.where(sliver, -0.3, c.h)
    lab, n = label(np.concatenate([c.solid, c.solid]))    # periodic in s
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        raise SystemExit(f"design has {n - 1} floating pieces (pixel sizes "
                         f"{sorted(sizes)[:-1]}); connect them before printing")
    thin = c.solid & ~binary_opening(c.solid, iterations=int(0.3 / RES))
    print(f"openwork check: connected, {thin.sum() * RES * RES:.2f} mm2 of strut "
          f"under 0.6 mm (curl tips)")
    return c


class OpenworkBand:
    def __init__(self, size, text="ALOHA", stone=0.0):
        self.ri = us_size_to_inner_diameter(size) / 2
        self.ro = self.ri + WALL
        self.rout = self.ro
        self.stone = stone
        c = check(compose(2 * np.pi * self.ro))
        self.c, self.circ = c, c.circ
        self.y0, self.n = c.y[0], c.solid.shape[0]

        tiled = np.concatenate([c.solid] * 3)
        sd = (distance_transform_edt(~tiled) - distance_transform_edt(tiled)) * RES
        self.sd = sd[self.n:2 * self.n]

        # outline between overlapping pieces, then the engraved lines
        edges = np.zeros(c.solid.shape, bool)
        edges[1:] |= c.label[1:] != c.label[:-1]
        edges[:, 1:] |= c.label[:, 1:] != c.label[:, :-1]
        edges = binary_dilation(edges & c.solid, iterations=1)
        h = c.h - np.maximum(c.cut, 0.09 * edges)
        self.h = gaussian_filter(np.where(c.solid, h, -0.3), 1.0)
        self.text = text_sdf_2d(text, TEXT_HEIGHT)
        self.z_top = self.ro + (1.6 if stone else 0.9)

    def sample(self, arr, s, y):
        fs = (s % self.circ) / RES
        fy = np.clip((y - self.y0) / RES, 0, arr.shape[1] - 1.001)
        i0 = fs.astype(int) % self.n
        i1 = (i0 + 1) % self.n
        j0 = fy.astype(int)
        a, b = fs - np.floor(fs), fy - j0
        return (arr[i0, j0] * (1 - a) * (1 - b) + arr[i1, j0] * a * (1 - b)
                + arr[i0, j0 + 1] * (1 - a) * b + arr[i1, j0 + 1] * a * b)

    def stone_setting(self, d, x, y, z, r):
        """Four-prong setting with a seat and light hole, at the flower centre."""
        rs = self.stone / 2
        rho = np.hypot(x, y)
        zz = r - self.ro                                # height above rail top
        z_g = 0.95                                      # stone girdle height
        collar = np.maximum(rho - (rs + 0.3), np.maximum(zz - (z_g - 0.05), -zz - 0.2))
        d = smin(d, collar, 0.15)
        for a in np.pi / 4 + np.arange(4) * np.pi / 2:
            px, py = x - (rs + 0.12) * np.cos(a), y - (rs + 0.12) * np.sin(a)
            prong = np.maximum(np.hypot(px, py) - 0.22, zz - (z_g + 0.42))
            tip = np.sqrt(px ** 2 + py ** 2 + (zz - (z_g + 0.42)) ** 2) - 0.22
            d = np.minimum(d, np.minimum(np.maximum(prong, -zz), tip))
        seat = np.where(zz >= z_g, rho - rs, rho - (rs - (z_g - zz) * 0.85))
        seat = np.maximum(seat, -(zz + 0.3))
        hole = rho - (rs - 0.3)                         # light hole, through
        return np.maximum(d, -np.minimum(seat, hole))

    def sdf(self, x, y, z):
        r = np.sqrt(x * x + z * z)
        phi = np.arctan2(x, z) % (2 * np.pi)
        s = phi * self.ro
        d2 = self.sample(self.sd, s, y)
        outer = r - (self.ro + self.sample(self.h, s, y))
        d = smax(outer, d2, 0.22)                         # softened edges
        d = np.maximum(d, smax(self.ri - r, d2, 0.3))     # comfort-fit inside
        if self.stone:                                    # top of the ring only
            d = np.where(z > 0, self.stone_setting(d, x, y, z, r), d)
        if self.text is not None:                         # inside, at the back
            si = -(phi - np.pi) * self.ri
            dt = np.maximum(self.text(si, y), r - (self.ri + TEXT_DEPTH))
            d = np.maximum(d, -dt)
        return np.maximum(d, self.ri - r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=7.0, help="US ring size")
    ap.add_argument("--stone", type=float, default=0.0,
                    help="round stone diameter for a centre setting, mm (0 = none)")
    ap.add_argument("--voxel", type=float, default=0.035, help="mesh resolution, mm")
    ap.add_argument("--tolerance", type=float, default=0.004,
                    help="max simplification deviation, mm")
    ap.add_argument("--text", default="ALOHA", help='inner engraving ("" for none)')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    band = OpenworkBand(args.size, args.text, args.stone)
    print(f"US size {args.size}: inner diameter {2*band.ri:.2f} mm")
    mesh = build_mesh(band, args.voxel, y_half=HW_FRONT + 0.3,
                      z_top=band.z_top, pad=0.4)
    print(f"marching cubes: {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
    mesh = decimate(mesh, args.tolerance)
    print(f"final: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume/1000:.3f} mL, bounds={np.round(mesh.extents, 2)} mm")
    stone = f"_stone{args.stone:g}mm" if args.stone else ""
    out = args.out or f"openwork_band_US{args.size:g}{stone}.stl"
    mesh.export(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
