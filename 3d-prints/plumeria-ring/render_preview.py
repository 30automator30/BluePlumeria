#!/usr/bin/env python3
"""Render shaded preview PNGs of a ring STL (point-splat z-buffer, no GPU)."""
import sys
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation

def rot(ax, a):
    c, s = np.cos(a), np.sin(a)
    if ax == "x": return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if ax == "y": return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def view(mesh, R, px=520, span=32.0, near_z=None):
    # sample dense points on the surface so splats leave no holes
    pts, fi = trimesh.sample.sample_surface_even(mesh, 2_500_000) if False else trimesh.sample.sample_surface(mesh, 3_000_000)
    nrm = mesh.face_normals[fi]
    p = pts @ R.T; n = nrm @ R.T
    if near_z is not None:
        keep = p[:, 2] < near_z; p, n = p[keep], n[keep]
    ix = ((p[:, 0] / span + 0.5) * px).astype(int)
    iy = ((0.5 - p[:, 1] / span) * px).astype(int)
    ok = (ix >= 0) & (ix < px) & (iy >= 0) & (iy < px)
    ix, iy, p, n = ix[ok], iy[ok], p[ok], n[ok]
    order = np.argsort(p[:, 2])          # far to near; nearest written last
    img = np.ones((px, px, 3)); depth = np.full((px, px), -1e9)
    L = np.array([-0.4, 0.5, 0.77]); L /= np.linalg.norm(L)
    lam = np.clip(n @ L, 0, 1); spec = np.clip(n @ L, 0, 1) ** 30
    shade = 0.25 + 0.7 * lam
    col = np.array([0.93, 0.80, 0.55])  # warm resin / gold tone
    c = np.clip(shade[:, None] * col + 0.35 * spec[:, None], 0, 1)
    o = order
    img[iy[o], ix[o]] = c[o]; depth[iy[o], ix[o]] = p[o, 2]
    holes = depth < -1e8
    for ch in range(3):
        dil = grey_dilation(np.where(holes, 0, img[..., ch]), size=2)
        img[..., ch] = np.where(holes & (grey_dilation(np.where(holes, -1e9, depth), size=3) > -1e8), dil, img[..., ch])
    return img

mesh = trimesh.load(sys.argv[1]); out = sys.argv[2]
views = [
    ("Top", rot("x", 0.0), None),
    ("3/4 view", rot("y", 0.5) @ rot("x", 0.75), None),
    ("Side", rot("z", np.pi/2) @ rot("y", -np.pi/2), None),
    ("Front (finger axis)", rot("x", -np.pi/2), None),
    ("Back", rot("x", np.pi), None),
    ("Inside back", rot("x", 0.0), 0.0),
]
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for a, (t, R, nz) in zip(axs.ravel(), views):
    if nz is not None:   # camera at ring centre looking at the inner back wall
        m = mesh.copy()
        img = view(m, R, span=20.0, near_z=nz)
    else:
        img = view(mesh, R)
    a.imshow(img); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=90)
print("saved", out)
