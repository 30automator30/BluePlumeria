"""A posed human figure from the MakeHuman base mesh (CC0; see mh/LICENSE.md).

The base mesh, the default skeleton and its skin weights live in mh/, with
the few body-shape targets this needs (young adult male). The figure is
shaped with those targets, then posed by aiming bones: each listed bone is
turned so it points along a given direction (and, for a few, so the body's
front faces a given way), its children carried along, and the skin follows
by linear blend skinning.

World frame after loading: z up, the figure facing -y, its left side +x,
units mm (scaled to a standing height).
"""

import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
MH = os.path.join(HERE, "mh")


def load_obj():
    """Vertices (all) and the triangles of the 'body' group."""
    vs, faces, grp = [], [], None
    with open(os.path.join(MH, "base.obj")) as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("g "):
                grp = line.split()[1]
            elif line.startswith("f ") and grp == "body":
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.array(vs), np.array(faces)


def apply_targets(v, muscle=0.75):
    """Young adult male, muscular, ideal proportions."""
    t = np.load(os.path.join(MH, "targets_male_young.npz"))
    v = v.copy()

    def add(name, w):
        if w <= 0:
            return
        idx, vec = t[name + ".index"], t[name + ".vector"] * 1e-3
        v[idx] += w * vec
    # muscle 0.5 = average, 1 = max
    m_max = max(0.0, (muscle - 0.5) * 2)
    add("universal-male-young-averagemuscle-averageweight", 1 - m_max)
    add("universal-male-young-maxmuscle-averageweight", m_max)
    add("proportions__male-young-averagemuscle-averageweight-idealproportions", 1 - m_max)
    add("proportions__male-young-maxmuscle-averageweight-idealproportions", m_max)
    return v


def to_world(v):
    """MakeHuman (y up, facing +z, left +x, decimetres) -> z up, facing -y."""
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


def rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    s, d = np.linalg.norm(c), a @ b
    if s < 1e-9:
        if d > 0:
            return np.eye(3)
        p = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    k = c / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = np.arctan2(s, d)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def frame_rot(a, fa, b, fb):
    """Rotation taking direction a to b and the front fa (made perpendicular
    to a) to fb (perpendicular to b)."""
    def fr(x, f):
        x = x / np.linalg.norm(x)
        f = f - (f @ x) * x
        f /= np.linalg.norm(f)
        return np.stack([x, f, np.cross(x, f)], 1)
    return fr(b, fb) @ fr(a, fa).T


class Figure:
    def __init__(self, height_mm=62.0, muscle=0.75):
        v, self.faces = load_obj()
        v = apply_targets(v, muscle)
        skel = json.load(open(os.path.join(MH, "default.mhskel")))
        self.bones = skel["bones"]
        joints = {k: v[np.array(ix)].mean(0) for k, ix in skel["joints"].items()}
        self.v_rest = to_world(v)
        self.joints = {k: to_world(p) for k, p in joints.items()}
        # scale to the standing height, feet on z = 0
        used = np.unique(self.faces)
        lo, hi = self.v_rest[used].min(0), self.v_rest[used].max(0)
        s = height_mm / (hi[2] - lo[2])
        off = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        self.v_rest = (self.v_rest + off) * s
        self.joints = {k: (p + off) * s for k, p in self.joints.items()}
        self.scale = s
        # skin weights, normalised per vertex
        W = json.load(open(os.path.join(MH, "default_weights.mhw")))["weights"]
        self.names = list(self.bones)
        self.bi = {n: i for i, n in enumerate(self.names)}
        rows, cols, vals = [], [], []
        for n, lst in W.items():
            if n not in self.bi:
                continue
            a = np.array(lst)
            rows += list(a[:, 0].astype(int))
            cols += [self.bi[n]] * len(a)
            vals += list(a[:, 1])
        from scipy.sparse import csr_matrix
        Wm = csr_matrix((vals, (rows, cols)), shape=(len(self.v_rest), len(self.names)))
        tot = np.asarray(Wm.sum(1)).ravel()
        tot[tot == 0] = 1
        from scipy.sparse import diags
        self.W = diags(1 / tot) @ Wm
        # hierarchy order
        order, seen = [], set()

        def visit(n):
            if n in seen:
                return
            p = self.bones[n].get("parent")
            if p:
                visit(p)
            seen.add(n)
            order.append(n)
        for n in self.names:
            visit(n)
        self.order = order
        self.children = {n: [] for n in self.names}
        for n in self.names:
            p = self.bones[n].get("parent")
            if p:
                self.children[p].append(n)

    def head(self, n):
        return self.joints[self.bones[n]["head"]]

    def tail(self, n):
        return self.joints[self.bones[n]["tail"]]

    def pose(self, aims, fronts=None, twists=None):
        """aims: {bone: world direction}; fronts: {bone: world direction the
        body's front (-y at rest) should face, for that bone}; twists:
        {bone: degrees about its own axis}. Bones are processed parent
        first; each one's rotation is applied about its posed head and
        carried down to its descendants. Returns the posed vertices and a
        dict of posed joint positions (bone heads and tails)."""
        fronts, twists = fronts or {}, twists or {}
        R = {n: np.eye(3) for n in self.names}          # global rotation of each bone
        Tt = {n: np.zeros(3) for n in self.names}       # global translation
        for n in self.order:
            p = self.bones[n].get("parent")
            if p:
                R[n], Tt[n] = R[p].copy(), Tt[p].copy()
            h = R[n] @ self.head(n) + Tt[n]
            d = R[n] @ (self.tail(n) - self.head(n))
            Q = np.eye(3)
            if n in aims:
                if n in fronts:
                    f = R[n] @ np.array([0, -1.0, 0])
                    Q = frame_rot(d, f, np.asarray(aims[n], float), np.asarray(fronts[n], float))
                else:
                    Q = rot_between(d, np.asarray(aims[n], float))
            if n in twists:
                ax = Q @ d
                ax /= np.linalg.norm(ax)
                a = np.radians(twists[n])
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                Q = (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K) @ Q
            R[n] = Q @ R[n]
            Tt[n] = h - R[n] @ self.head(n)
        # linear blend skinning
        V = np.zeros_like(self.v_rest)
        Wc = self.W.tocsc()
        for n in self.names:
            col = Wc[:, self.bi[n]]
            if col.nnz == 0:
                continue
            ix, w = col.indices, col.data
            V[ix] += w[:, None] * (self.v_rest[ix] @ R[n].T + Tt[n])
        J = {}
        for n in self.names:
            J[n] = (R[n] @ self.head(n) + Tt[n], R[n] @ self.tail(n) + Tt[n])
        return V, J

    def mesh(self, V):
        m = trimesh.Trimesh(V, self.faces, process=False)
        m.remove_unreferenced_vertices()
        return m


def ik_dir(S, T, L1, L2, pole):
    """Directions of the two bones of a limb from S reaching for T, the
    middle joint bending toward pole; T is pulled in if out of reach."""
    D = T - S
    dl = min(np.linalg.norm(D), (L1 + L2) * 0.995)
    Dn = D / np.linalg.norm(D)
    a = (L1 ** 2 - L2 ** 2 + dl ** 2) / (2 * dl)
    h = np.sqrt(max(L1 ** 2 - a ** 2, 0))
    p = pole - (pole @ Dn) * Dn
    p /= np.linalg.norm(p)
    M = S + Dn * a + p * h
    E = S + Dn * dl
    return (M - S) / np.linalg.norm(M - S), (E - M) / np.linalg.norm(E - M)


def limb_len(F, a, b):
    """Rest length from the head of bone a to the head of bone b."""
    return np.linalg.norm(F.head(b) - F.head(a))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)
