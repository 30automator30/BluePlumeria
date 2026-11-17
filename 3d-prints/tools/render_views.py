#!/usr/bin/env python3
"""Shaded multi-view preview renders of one or more STLs (point-splat
z-buffer; no GPU or display needed). This is what the approval previews are
made with.

  python3 render_views.py OUT.png part.stl [part2.stl ...]
      [--colors 0.62,0.86,0.90 0.80,0.52,0.30]  one RGB per part
      [--up z|y]          z: sculptures standing on the plate (default)
                          y: flat pieces drawn in the xy plane (suncatchers)
      [--span 170]        width of each view, mm
      [--close=X,Y,Z --close-span 60]  (use = when X is negative)   add a close-up around a point
      [--size 7]          inches per panel

Views: front, three-quarter from each side (or tilted for --up y), back,
and edge-on for flat pieces; plus the optional close-up.
"""
import argparse
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import grey_dilation  # noqa: E402



def rot(ax, a):
    c, s = np.cos(a), np.sin(a)
    if ax == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def view(parts, R, px=700, span=140, centre=(0, 0, 0), n=6_000_000):
    """parts: [(mesh, rgb)]. R rotates world to view (x right, y up, z toward
    the viewer)."""
    P, N, C = [], [], []
    tot = sum(m.area for m, _ in parts)
    for m, col in parts:
        k = int(n * m.area / tot) + 1000
        pts, fi = trimesh.sample.sample_surface(m, k)
        P.append(pts - np.asarray(centre))
        N.append(m.face_normals[fi])
        C.append(np.tile(col, (k, 1)))
    p = np.vstack(P) @ R.T
    nn = np.vstack(N) @ R.T
    col = np.vstack(C)
    ix = ((p[:, 0] / span + 0.5) * px).astype(int)
    iy = ((0.5 - p[:, 1] / span) * px).astype(int)
    ok = (ix >= 0) & (ix < px) & (iy >= 0) & (iy < px)
    ix, iy, p, nn, col = ix[ok], iy[ok], p[ok], nn[ok], col[ok]
    o = np.argsort(p[:, 2])
    img = np.ones((px, px, 3))
    depth = np.full((px, px), -1e9)
    L = np.array([-0.4, 0.5, 0.77])
    L /= np.linalg.norm(L)
    lam = np.clip(nn @ L, 0, 1)
    c = np.clip((0.25 + 0.7 * lam)[:, None] * col + 0.35 * (lam ** 30)[:, None], 0, 1)
    img[iy[o], ix[o]] = c[o]
    depth[iy[o], ix[o]] = p[o, 2]
    holes = depth < -1e8
    for ch in range(3):
        dil = grey_dilation(np.where(holes, 0, img[..., ch]), size=2)
        near = grey_dilation(np.where(holes, -1e9, depth), size=3) > -1e8
        img[..., ch] = np.where(holes & near, dil, img[..., ch])
    return img


def panel(parts, views, out, ncol=2, size=7):
    nrow = (len(views) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow), squeeze=False)
    for a in axs.ravel():
        a.axis("off")
    for a, (title, R, span, centre) in zip(axs.ravel(), views):
        a.imshow(view(parts, R, span=span, centre=centre))
        a.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=80)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("stls", nargs="+")
    ap.add_argument("--colors", nargs="*", default=[])
    ap.add_argument("--up", choices=["z", "y"], default="z")
    ap.add_argument("--span", type=float, default=None)
    ap.add_argument("--close", default=None)
    ap.add_argument("--close-span", type=float, default=60)
    ap.add_argument("--size", type=float, default=7)
    a = ap.parse_args()
    default = [(0.62, 0.86, 0.90), (0.80, 0.52, 0.30), (0.93, 0.80, 0.55)]
    parts = []
    for i, f in enumerate(a.stls):
        col = np.array([float(t) for t in a.colors[i].split(",")]) if i < len(a.colors) else np.array(default[i % 3])
        parts.append((trimesh.load(f), col))
    lo = np.min([m.bounds[0] for m, _ in parts], 0)
    hi = np.max([m.bounds[1] for m, _ in parts], 0)
    c = tuple((lo + hi) / 2)
    span = a.span or float(np.max(hi - lo) * 1.12)
    if a.up == "z":
        views = [("front", rot("x", -np.pi / 2 + 0.12), span, c),
                 ("3/4 from front-left", rot("x", -np.pi / 2 + 0.3) @ rot("z", 0.6), span, c),
                 ("3/4 from front-right", rot("x", -np.pi / 2 + 0.3) @ rot("z", -0.6), span, c),
                 ("back", rot("x", -np.pi / 2) @ rot("z", np.pi), span, c)]
    else:
        views = [("front", rot("x", 0.0), span, c),
                 ("tilted", rot("x", -0.6) @ rot("z", 0.3), span, c),
                 ("back", rot("y", np.pi), span, c),
                 ("edge-on", rot("y", np.pi / 2) @ rot("x", -0.15), max(20.0, (hi - lo)[2] * 8), c)]
    if a.close:
        p = tuple(float(t) for t in a.close.split(","))
        R = rot("x", -0.5) @ rot("z", 0.25) if a.up == "y" else rot("x", -np.pi / 2 + 0.3) @ rot("z", 0.4)
        views.append(("close-up", R, a.close_span, p))
    panel(parts, views, a.out, size=a.size)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
