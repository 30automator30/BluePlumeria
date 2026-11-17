#!/usr/bin/env python3
"""Hawaiian heirloom-style band generator (resin / MSLA printing).

A wide band in the style of hand-engraved Hawaiian heirloom jewelry:
raised plumerias, curling wave scrolls and hatched leaves stand proud of a
recessed, stippled background, and every open space is filled with small
curls, leaves and engraver's sweeps; each element is bevelled toward its
engraved outlines, and both edges are scalloped.

The motif is drawn as a 2D relief height map on the unrolled band
(s = arc length, y = across the band), wrapped onto the ring as a signed
distance field, meshed with marching cubes and written as a binary STL.

Usage
  python3 heirloom_band.py --size 7
  python3 heirloom_band.py --size 7 --text "MAHALO"
  python3 heirloom_band.py --size 7 --width 6      # narrower band
"""

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "plumeria-ring"))
from plumeria_ring import (build_mesh, decimate, petal_shape, smin,  # noqa: E402
                           text_sdf_2d, us_size_to_inner_diameter)

RES = 0.02                 # height-map resolution, mm/pixel
WALL = 1.55                # band thickness at the raised design surface
RELIEF = 0.40              # background depth below the design
DOME = 0.20                # bevel height of each raised element
GROOVE = 0.06              # half width of engraved lines
ENGRAVE = 0.14             # extra depth of engraved lines below the bevel
TEXT_HEIGHT, TEXT_DEPTH = 1.6, 0.25
SCALLOP_PITCH, SCALLOP_DEPTH = 2.6, 0.45   # edge scallop spacing and notch depth


def smax(a, b, k):
    return -smin(-a, -b, k)


# ------------------------------------------------------------- 2D artwork

class Motif:
    """Draws one repeat of the design on an (s, y) pixel grid."""

    def __init__(self, unit, half_w):
        self.unit, self.hw = unit, half_w
        self.s = np.arange(0, unit, RES)
        self.y = np.arange(-half_w - 0.3, half_w + 0.3, RES)
        self.S, self.Y = np.meshgrid(self.s, self.y, indexing="ij")
        self.raised = np.zeros(self.S.shape, bool)
        self.groove = np.zeros(self.S.shape, bool)
        self.label = np.zeros(self.S.shape, np.int16)
        self.next_label = 1

    def local(self, s0, y0, ang=0.0):
        ds = (self.S - s0 + self.unit / 2) % self.unit - self.unit / 2   # wrap
        dy = self.Y - y0
        ca, sa = np.cos(ang), np.sin(ang)
        return ds * ca + dy * sa, -ds * sa + dy * ca

    def paint(self, mask, behind=False):
        """Add a raised element on top of (or tucked behind) what is drawn."""
        if behind:
            mask = mask & ~self.raised
        self.raised |= mask
        self.label[mask] = self.next_label
        self.next_label += 1
        return mask

    def plumeria(self, s0, y0, radius, rot):
        k = radius / 7.5
        x, y = self.local(s0, y0)
        x, y = x / k, y / k
        petals = []
        for i in range(5):
            th = rot + 2 * np.pi * i / 5
            dx, dy = np.cos(th), np.sin(th)
            px, py = x + 1.0 * dy, y - 1.0 * dx
            u, v = px * dx + py * dy, -px * dy + py * dx
            t = np.clip(u / 7.5, 0, 1)
            w = 3.3 * petal_shape(t) * np.where(v > 0, 1.1, 0.9)
            petals.append((np.abs(v) < w) & (u > 0) & (u < 7.5))
            # centre crease running most of the way up each petal
            self.groove |= (np.abs(v - 0.12 * u) < GROOVE / k * 0.6) & (t > 0.18) & (t < 0.5) \
                & petals[-1]
        for m in petals:                  # pinwheel: each petal over the last
            self.paint(m)
        rr = np.hypot(x, y)
        self.paint(rr < 1.3)
        self.groove |= np.abs(rr - 1.3) < GROOVE / k
        self.groove |= rr < 0.45

    def scroll(self, s0, y0, radius, turns=2.6, sense=1, behind=False):
        """Curling ocean-wave scroll: a spiral groove with combed hatching."""
        x, y = self.local(s0, y0)
        y = y * sense
        rho, th = np.hypot(x, y), np.arctan2(y, x) % (2 * np.pi)
        pitch = radius / turns
        a = pitch * th / (2 * np.pi)
        dist = np.abs(((rho - a + pitch / 2) % pitch) - pitch / 2)
        inside = self.paint(rho < radius, behind)
        self.groove |= inside & (dist < GROOVE) & (rho > 0.25)
        # comb hatching on the outer whorl, like a breaking wave
        outer = inside & (rho > radius - pitch * 0.95) & (dist > GROOVE * 2.2)
        comb = np.abs(((th * radius / 0.26) % 1.0) - 0.5) < 0.16
        self.groove |= outer & comb

    def ribbon(self, s_from, s_to, y_from, y_to, width, sag):
        """Raised wave ribbon with diagonal hatching; returns its centreline."""
        span = (s_to - s_from) % self.unit

        def centre(u):
            t = np.clip(u / span, 0, 1)
            return y_from + (y_to - y_from) * (3 * t ** 2 - 2 * t ** 3) + sag * np.sin(np.pi * t)

        u = (self.S - s_from) % self.unit
        t = np.clip(u / span, 0, 1)
        yc = centre(u)
        taper = width * (0.45 + 0.55 * np.sin(np.pi * np.clip(t * 0.85 + 0.15, 0, 1)))
        m = (u < span) & (np.abs(self.Y - yc) < taper)
        self.paint(m)
        hatch = np.abs((((u + 0.7 * (self.Y - yc)) / 0.3) % 1.0) - 0.5) < 0.17
        self.groove |= m & hatch & (np.abs(self.Y - yc) < taper - 0.12)
        return lambda s: centre((s - s_from) % self.unit)

    def leaf(self, s0, y0, ang, length, half_w, behind=False):
        """Lanceolate leaf: midrib plus hatching on one half."""
        u, v = self.local(s0, y0, ang)
        t = np.clip(u / length, 0, 1)
        w = half_w * np.sin(np.pi * t ** 0.75) ** 0.8
        m = self.paint((u > 0) & (u < length) & (np.abs(v) < w), behind)
        self.groove |= m & (np.abs(v) < GROOVE * 0.8) & (t > 0.08) & (t < 0.92)
        hatch = np.abs((((u - 0.9 * v) / 0.24) % 1.0) - 0.5) < 0.17
        self.groove |= m & hatch & (v > GROOVE * 2.2) & (v < w - 0.08)

    def sweep(self, s0, y0, radius, ang, behind=False):
        """Crescent sweep, the engraver's swoosh, with a groove along its arc."""
        x, y = self.local(s0, y0, ang)
        rho = np.hypot(x, y)
        m = (rho < radius) & (np.hypot(x - 0.45 * radius, y) > 0.85 * radius)
        m = self.paint(m, behind)
        if radius > 0.7:
            self.groove |= m & (np.abs(rho - 0.72 * radius) < GROOVE * 0.8)

    def free_radius(self):
        """Distance (mm) from each pixel to the nearest drawn element or edge."""
        occ = self.raised | (np.abs(self.Y) > self.hw - 0.2)
        n = occ.shape[0]
        edt = distance_transform_edt(~np.concatenate([occ, occ, occ])) * RES
        return edt[n:2 * n]

    def fill_gaps(self, min_r=0.45, seed=3):
        """Fill empty background with curls, leaves and sweeps, biggest first."""
        rng = np.random.default_rng(seed)
        for _ in range(200):
            e = self.free_radius()
            i, j = np.unravel_index(np.argmax(e), e.shape)
            r = e[i, j]
            if r < min_r:
                break
            s0, y0 = self.s[i], self.y[j]
            n = e.shape[0]
            j0, j1 = max(j - 3, 0), min(j + 3, e.shape[1] - 1)
            gs = e[(i + 3) % n, j] - e[(i - 3) % n, j]
            gy = e[i, j1] - e[i, j0]
            away = np.arctan2(gy, gs)            # direction of most room
            if r > 1.1:
                self.scroll(s0, y0, 0.95 * r, turns=1.7,
                            sense=rng.choice([-1, 1]), behind=True)
            elif r > 0.72:
                ang = away + np.pi / 2 + rng.uniform(-0.35, 0.35)
                ln = 2.7 * r
                self.leaf(s0 - np.cos(ang) * ln / 2, y0 - np.sin(ang) * ln / 2,
                          ang, ln, 0.62 * r, behind=True)
            else:
                self.sweep(s0, y0, 1.15 * r, away + np.pi, behind=True)

    def height(self, seed=7):
        """Relief height (mm, 0 = design surface) and edge-scallop profile."""
        edges = np.zeros(self.S.shape, bool)
        edges[1:] |= self.label[1:] != self.label[:-1]
        edges[:, 1:] |= self.label[:, 1:] != self.label[:, :-1]
        edges &= self.raised | np.roll(self.raised, 1, 0) | np.roll(self.raised, 1, 1)
        edges = binary_dilation(edges, iterations=max(1, int(GROOVE / RES) - 1))
        cut = (self.groove | edges) & self.raised

        body = self.raised & ~cut
        edt = distance_transform_edt(body) * RES
        h = np.where(body, DOME * np.clip(edt / 0.35, 0, 1) ** 0.7 - DOME, 0.0)
        h = np.where(cut, -DOME - ENGRAVE, h)

        rng = np.random.default_rng(seed)       # stippled, matte background
        stip = gaussian_filter(rng.random(self.S.shape), 1.2)
        stip = (stip - stip.mean()) / stip.std()
        h = np.where(self.raised, h, -RELIEF - 0.03 * stip)
        h = gaussian_filter(h, 0.8)

        # scalloped edges: a row of shallow arcs along each edge, with the
        # top and bottom rows offset half a scallop
        n = max(1, round(self.unit / SCALLOP_PITCH))

        def edge_profile(sign):
            f = (self.s / (self.unit / n) + (0.5 if sign < 0 else 0.0)) % 1.0
            arc = np.sqrt(np.clip(1 - (2 * f - 1) ** 2, 0, 1))
            return self.hw - SCALLOP_DEPTH * (1 - arc) ** 0.8

        return h, edge_profile(1), edge_profile(-1)


def draw_unit(unit, half_w):
    """One design repeat: plumeria -> wave scroll -> vine with leaf spray."""
    m = Motif(unit, half_w)
    k = unit / 21.5
    q = half_w / 4.0
    # the vine is drawn first in a scratch motif just to get its centreline
    scratch = Motif(unit, half_w)
    vine = scratch.ribbon(9.0 * k, unit - 0.4, -0.45 * q, -2.75 * q, 0.55 * q, -0.45 * q)

    # leaves sprouting up and to the right off the vine, filling to the edge
    for f in [11.6, 13.4, 15.2, 17.0, 18.8]:
        sb = f * k
        yb = float(vine(np.array(sb))) + 0.15
        ang = 1.0
        length = min(3.6 * q, (half_w - 0.25 - yb) / np.sin(ang) + 0.6)
        m.leaf(sb, yb, ang, length, 0.78 * q)
    # leaves curling down off the vine into the lower spaces
    for f, ang, ln in [(11.0, -2.35, 2.9), (13.6, -2.2, 2.0), (16.6, -0.55, 1.4)]:
        sb = f * k
        m.leaf(sb, float(vine(np.array(sb))) - 0.1, ang, ln * q, 0.6 * q)
    m.ribbon(9.0 * k, unit - 0.4, -0.45 * q, -2.75 * q, 0.55 * q, -0.45 * q)
    m.scroll(10.2 * k, 1.35 * q, 2.05 * q)
    # foliage tucked behind the plumeria
    m.leaf(5.9 * k, 2.55 * q, 0.15, 2.7 * q, 0.55 * q)
    m.leaf(6.4 * k, -2.65 * q, -0.1, 2.4 * q, 0.5 * q)
    m.plumeria(3.9 * k, 0.0, 3.6 * q, 0.35)
    m.fill_gaps()                     # no bare background: leaves and sweeps
    return m


# ------------------------------------------------------------------- ring

class HeirloomBand:
    def __init__(self, size, width, text):
        self.ri = us_size_to_inner_diameter(size) / 2
        self.ro = self.ri + WALL
        self.hw = width / 2
        circ = 2 * np.pi * self.ro
        n_units = max(2, round(circ / 21.5))
        self.unit = circ / n_units
        m = draw_unit(self.unit, self.hw)
        self.h, self.top, self.bot = m.height()
        self.y0 = m.y[0]
        self.text = text_sdf_2d(text, TEXT_HEIGHT)
        # for build_mesh's bounding box
        self.rout = self.ro

    def sample(self, s, y):
        fs = (s % self.unit) / RES
        fy = np.clip((y - self.y0) / RES, 0, self.h.shape[1] - 1.001)
        i0 = fs.astype(int) % self.h.shape[0]
        i1 = (i0 + 1) % self.h.shape[0]
        j0 = fy.astype(int)
        a, b = fs - np.floor(fs), fy - j0
        h = self.h
        hv = (h[i0, j0] * (1 - a) * (1 - b) + h[i1, j0] * a * (1 - b)
              + h[i0, j0 + 1] * (1 - a) * b + h[i1, j0 + 1] * a * b)
        top = self.top[i0] * (1 - a) + self.top[i1] * a
        bot = self.bot[i0] * (1 - a) + self.bot[i1] * a
        return hv, top, bot

    def sdf(self, x, y, z):
        r = np.sqrt(x * x + z * z)
        phi = np.arctan2(x, z) % (2 * np.pi)
        s = phi * self.ro
        h, top, bot = self.sample(s, y)
        outer = r - (self.ro + h)
        edge = np.where(y > 0, y - top, -y - bot)
        d = smax(outer, edge, 0.35)                     # rounded outer edges
        d = np.maximum(d, smax(self.ri - r, edge, 0.5))  # comfort-fit inside
        if self.text is not None:                       # inside, at the back
            si = -(phi - np.pi) * self.ri
            dt = np.maximum(self.text(si, y), r - (self.ri + TEXT_DEPTH))
            d = np.maximum(d, -dt)
        return np.maximum(d, self.ri - r)


def set_metal():
    """Wider, deeper engraving that survives casting and plating."""
    global GROOVE, ENGRAVE, RELIEF, TEXT_DEPTH
    GROOVE, ENGRAVE, RELIEF, TEXT_DEPTH = 0.09, 0.30, 0.55, 0.40


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=7.0, help="US ring size")
    ap.add_argument("--width", type=float, default=8.0, help="band width, mm")
    ap.add_argument("--voxel", type=float, default=0.04, help="mesh resolution, mm")
    ap.add_argument("--tolerance", type=float, default=0.004,
                    help="max simplification deviation, mm")
    ap.add_argument("--text", default="ALOHA", help='inner engraving ("" for none)')
    ap.add_argument("--fit", type=float, default=0.5,
                    help="mm added to the inner diameter to offset resin shrinkage "
                         "(0.5 matched test prints; add ~0.3 more for casting in silver)")
    ap.add_argument("--metal", action="store_true",
                    help="for casting or electroplating: thicker wires, deeper engravings")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import plumeria_ring as _pr
    _pr.FIT_ALLOWANCE = args.fit
    if args.metal:
        set_metal()

    band = HeirloomBand(args.size, args.width, args.text)
    print(f"US size {args.size}: inner diameter {2*band.ri:.2f} mm, "
          f"{round(2*np.pi*band.ro/band.unit)} design repeats")
    mesh = build_mesh(band, args.voxel, y_half=band.hw + 0.4,
                      z_top=band.ro + 0.4, pad=0.4)
    print(f"marching cubes: {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
    mesh = decimate(mesh, args.tolerance)
    print(f"final: {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume/1000:.3f} mL, bounds={np.round(mesh.extents, 2)} mm")
    out = args.out or f"heirloom_band_US{args.size:g}{'_metal' if args.metal else ''}.stl"
    mesh.export(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
