"""The surfer: the MakeHuman figure (see human.py) posed in a deep, driving
crouch, chest to the viewer, riding toward +x.

Deck frame: origin on the board's centre line, between the feet; x toward
the board's nose; z up (the figure stands plumb); the deck tilts down
toward -y (the viewer) by TILT, as a board does on its rail on a face.
"""

import numpy as np

from human import Figure, ik_dir, limb_len, unit

HEIGHT = 62.0           # standing height, mm
STANCE = 25.0           # between the ankles, along the board
HIP_H = 24.0            # hips above the deck in the crouch


def pose(tilt_deg, height=HEIGHT):
    F = Figure(height_mm=height, muscle=0.8)
    t = np.radians(tilt_deg)
    n = np.array([0, -np.sin(t), np.cos(t)])        # deck normal
    down = np.array([0, -np.cos(t), -np.sin(t)])     # down the deck, across the board (toward the viewer)
    s = height / 62.0
    # the deck sits below the figure's rest hips so the hips end up HIP_H above it
    hip_rest = F.head("upperleg01.L")[2]
    z0 = hip_rest - HIP_H * s

    def on_deck(x, across):
        """A point on the deck, `across` mm downhill (toward the viewer) of
        the centre line."""
        return np.array([x, 0, z0]) + down * across

    # torso: lean toward the nose and a little toward the viewer, chest turned
    # partly down the line
    aims = {                                   # a hip hinge: the back flat, leaning out over the knees
        "root": unit([0.12, -0.86, -0.50]),      # the pelvis tips forward with it, so the seat doesn't bulge
        "spine05": unit([0.22, -0.48, 1.0]),
        "spine04": unit([0.30, -0.52, 1.0]),
        "spine03": unit([0.36, -0.50, 1.0]),
        "spine02": unit([0.40, -0.45, 1.0]),
        "spine01": unit([0.42, -0.38, 1.0]),
        "neck01": unit([0.35, -0.22, 1.0]),
        "neck02": unit([0.25, -0.08, 1.0]),
        "neck03": unit([0.15, 0.0, 1.0]),
        "head": unit([0.10, 0.05, 1.0]),
    }
    fronts = {
        "spine05": unit([0.06, -1, 0]), "spine04": unit([0.09, -1, 0]), "spine03": unit([0.12, -1, 0]),
        "spine02": unit([0.15, -1, 0]), "spine01": unit([0.18, -1, 0]),
        "neck01": unit([0.6, -1, 0]), "neck02": unit([0.9, -1, 0]), "neck03": unit([1.2, -1, 0]),
        "head": unit([1.5, -0.8, -0.25]),
    }
    _, J = F.pose(aims, fronts)

    # legs by IK: hips where the posed spine left them, ankles on the deck
    ank_h = 2.6 * s
    for side, x, pole, toe in (("L", STANCE / 2, unit([0.7, -0.7, 0.1]), unit([0.35, -1, 0])),
                               ("R", -STANCE / 2, unit([0.15, -1.0, -0.35]), unit([0.12, -1, 0]))):
        S = J[f"upperleg01.{side}"][0]
        L1 = limb_len(F, f"upperleg01.{side}", f"lowerleg01.{side}")
        L2 = limb_len(F, f"lowerleg01.{side}", f"foot.{side}")
        T = on_deck(x, -0.6 * s) + n * ank_h
        d1, d2 = ik_dir(S, T, L1, L2, pole)
        aims[f"upperleg01.{side}"] = d1
        aims[f"upperleg02.{side}"] = d1
        aims[f"lowerleg01.{side}"] = d2
        aims[f"lowerleg02.{side}"] = d2
        # foot flat on the deck, toes downhill toward the viewer, turned a little toward the nose
        ft = toe - (toe @ n) * n
        aims[f"foot.{side}"] = unit(ft - 0.35 * n)
    # arms by IK: front (left) arm reaching forward and low down the line;
    # back (right) arm out behind, toward the wave
    for side, reach, pole in (("L", np.array([14.5, -6.0, -10.0]), unit([0.2, 0.3, -1.0])),
                              ("R", np.array([-12.5, 8.5, -3.5]), unit([-0.2, 0.2, -1.0]))):
        S = J[f"upperarm01.{side}"][0]
        L1 = limb_len(F, f"upperarm01.{side}", f"lowerarm01.{side}")
        L2 = limb_len(F, f"lowerarm01.{side}", f"wrist.{side}")
        d1, d2 = ik_dir(S, S + reach * s, L1, L2, pole)
        for b in ("upperarm01", "upperarm02"):
            aims[f"{b}.{side}"] = d1
        for b in ("lowerarm01", "lowerarm02"):
            aims[f"{b}.{side}"] = d2
        aims[f"wrist.{side}"] = unit(d2 + np.array([0, 0, -0.15]))
    V, J = F.pose(aims, fronts)
    # move so the deck centre between the feet is the origin
    off = np.array([0, 0, z0])
    V = V - off
    J = {k: (a - off, b - off) for k, (a, b) in J.items()}
    # smooth out the creases skinning leaves where the spine and hips bend
    # hard (Taubin smoothing, which keeps volume), on the torso, hips and
    # thighs only; face, hands and feet keep their detail
    tors = [i for k, i in F.bi.items() if k.startswith(("spine", "root", "pelvis", "upperleg", "breast", "clavicle"))]
    knee = [i for k, i in F.bi.items() if k.startswith("lowerleg01")]
    wmask = np.clip(np.asarray(F.W[:, tors].sum(1)).ravel() + 0.6 * np.asarray(F.W[:, knee].sum(1)).ravel(), 0, 1)
    V = taubin(V, F.faces, wmask, iters=12)
    chest = [i for k, i in F.bi.items() if k.startswith(("breast", "spine02", "spine01"))]
    cm = np.clip(np.asarray(F.W[:, chest].sum(1)).ravel() * 1.5, 0, 1)
    cm *= (F.v_rest[:, 1] / s < 2.5)                                  # the front of the chest only
    V = taubin(V, F.faces, cm, iters=40)
    # where the clothes and hair go, decided on the standing figure
    R = F.v_rest / s
    knee, hip = 17.0, 32.7
    x, y, z = R[:, 0], R[:, 1], R[:, 2]

    def ramp(a, w=0.35):
        return np.clip(a / w + 0.5, 0, 1)
    # board shorts: from the waist to a hem a little above mid-thigh (the hem
    # dips at the outside of the leg); not the arms or hands
    hem = knee + 0.38 * (hip - knee) - 0.8 * np.clip((np.abs(x) - 4.0) / 3.0, 0, 1)
    shorts = ramp(z - hem, 0.15) * ramp(35.4 - z, 0.15) * ramp(10.5 - np.abs(x), 1.0)
    # fabric: folds round the thighs and up from the crotch, cut into the cloth
    folds = np.abs(np.sin(np.pi * (z + 0.7 * np.sin(1.3 * x) + 0.5 * np.abs(x)) / 1.9)) ** 10
    folds *= np.clip((31.0 - z) / 3.0, 0, 1)
    band = np.clip(1 - np.abs(z - 34.9) / 0.5, 0, 1)                 # the waistband
    cloth = shorts * (1.0 - 0.45 * folds + 0.3 * band)
    # hair: above a hairline that sweeps from the brow down behind the ears
    # to the nape
    line = 59.4 + 0.5 * np.abs(x) / 4.0 - 2.6 * np.clip((y - 1.5) / 3.5, 0, 1)
    hair = np.clip((z - line) / 2.2, 0, 1) ** 0.7 * ramp(6.2 - np.abs(x), 0.8)   # thickening up from the hairline
    # board shorts as cloth. Subdivide twice first so the cloth is smooth
    # on the coarse base mesh; smooth the skin under the shorts hard so the
    # cloth spans the seat and the crotch; then lay the cloth on, 0.9 mm
    # thick with soft folds, a waistband and a crisp hem
    import trimesh
    used = np.unique(F.faces)
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    faces = remap[F.faces]
    V, at = V[used], np.stack([shorts[used], hair[used], cloth[used]], 1)
    for _ in range(2):
        V, faces, attrs = trimesh.remesh.subdivide(V, faces, vertex_attributes=dict(a=at))
        at = attrs["a"]
    shorts, hair, cloth = at[:, 0], at[:, 1], at[:, 2]
    cloth = smooth_attr(cloth, faces, iters=6) * (shorts > 0.02)
    V = V + vertex_normals(V, faces) * (1.0 * s * cloth)[:, None]
    return dict(V=V, faces=faces, J=J, shorts=shorts, hair=hair, F=F)


def smooth_attr(a, faces, iters=6):
    from scipy.sparse import coo_matrix, diags
    n = len(a)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    P = diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
    for _ in range(iters):
        a = 0.5 * a + 0.5 * (P @ a)
    return a


def vertex_normals(V, faces):
    fn = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / (np.linalg.norm(n, axis=1)[:, None] + 1e-12)


def taubin(V, faces, wmask, iters=12, lam=0.5, mu=-0.53):
    from scipy.sparse import coo_matrix, diags
    n = len(V)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    P = diags(1 / deg) @ A
    W = wmask[:, None]
    V = V.copy()
    for _ in range(iters):
        V = V + lam * W * (P @ V - V)
        V = V + mu * W * (P @ V - V)
    return V


if __name__ == "__main__":
    import sys
    P = pose(22.0)
    import trimesh
    m = trimesh.Trimesh(P["V"], P["faces"])
    print(len(m.faces), m.is_watertight, np.round(m.bounds, 1))
    m.export(sys.argv[1])
