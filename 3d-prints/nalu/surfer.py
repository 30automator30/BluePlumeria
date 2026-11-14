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
HIP_H = 21.0            # hips above the deck in the crouch


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
    aims = {
        "spine05": unit([0.18, -0.10, 1.0]),
        "spine04": unit([0.30, -0.14, 1.0]),
        "spine03": unit([0.42, -0.18, 1.0]),
        "spine02": unit([0.50, -0.20, 1.0]),
        "spine01": unit([0.55, -0.20, 1.0]),
        "neck01": unit([0.45, -0.22, 1.0]),
        "neck02": unit([0.30, -0.20, 1.0]),
        "neck03": unit([0.20, -0.18, 1.0]),
        "head": unit([0.12, -0.12, 1.0]),
    }
    fronts = {
        "spine05": unit([0.10, -1, 0]), "spine04": unit([0.18, -1, 0]), "spine03": unit([0.26, -1, 0]),
        "spine02": unit([0.32, -1, 0]), "spine01": unit([0.36, -1, 0]),
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
    # where the clothes and hair go, decided on the standing figure
    R = F.v_rest / s
    knee, hip = 17.0, 32.7
    x, y, z = R[:, 0], R[:, 1], R[:, 2]

    def ramp(a, w=0.35):
        return np.clip(a / w + 0.5, 0, 1)
    # board shorts: from the waist to a hem a little above mid-thigh (the hem
    # dips at the outside of the leg); not the arms or hands
    hem = knee + 0.45 * (hip - knee) - 0.8 * np.clip((np.abs(x) - 4.0) / 3.0, 0, 1)
    shorts = ramp(z - hem) * ramp(35.4 - z) * ramp(10.5 - np.abs(x), 1.0)
    # hair: above a hairline that sweeps from the brow down behind the ears
    # to the nape
    line = 59.4 + 0.5 * np.abs(x) / 4.0 - 2.6 * np.clip((y - 1.5) / 3.5, 0, 1)
    hair = np.clip((z - line) / 2.2, 0, 1) ** 0.7 * ramp(6.2 - np.abs(x), 0.8)   # thickening up from the hairline
    return dict(V=V, faces=F.faces, J=J, shorts=shorts, hair=hair, F=F)


if __name__ == "__main__":
    import sys
    P = pose(22.0)
    m = P["F"].mesh(P["V"])
    print(len(m.faces), m.is_watertight, np.round(m.bounds, 1))
    m.export(sys.argv[1])
