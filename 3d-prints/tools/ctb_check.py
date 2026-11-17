#!/usr/bin/env python3
"""Read a CHITUBOX .ctb (encrypted CTB v4/v5, as CHITUBOX writes for the
Saturn 4 Ultra): print its settings and check every layer for islands.

The decryption follows the scheme documented in the open-source UVtools
project: the settings block is AES-256-CBC, each layer's RLE data is XORed
with a per-layer key stream (and may have an AES-encrypted span).

  python3 ctb_check.py file.ctb                 # settings + island check
  python3 ctb_check.py file.ctb --settings      # settings only
  python3 ctb_check.py file.ctb --ds 8 --json out.json

Islands are counted the way the slicer sees them: a downsampled grid
(--ds pixels per cell; 8 px is about 0.15 mm on the Saturn 4 Ultra),
8-connected pieces, and a piece rests on the layer below if any of its
8 neighbours was cured there. A pixel counts as solid above 50% grey. Judge
each island by how many layers it floats before joining (reported): most
join within 1-2 layers and are harmless; flat bottoms between support tips
are the real ones.
"""
import argparse
import base64
import json
import struct

import numpy as np
from scipy import ndimage

_K = "UVtools"


def _x(bs):
    return bytes(c ^ ord(_K[i % len(_K)]) for i, c in enumerate(bs))


KEY = _x(base64.b64decode("hQ36XB6yTk+zO02ysyiowt8yC1buK+nbLWyfY40EXoU="))
IV = _x(base64.b64decode("Wld+ampndVJecmVjYH5cWQ=="))
M32 = 0xFFFFFFFF

FIELDS = """Q ChecksumValue
I LayerPointersOffset
f DisplayWidth
f DisplayHeight
f MachineZ
I Unknown1
I Unknown2
f TotalHeightMillimeter
f LayerHeight
f ExposureTime
f BottomExposureTime
f LightOffDelay
I BottomLayerCount
I ResolutionX
I ResolutionY
I LayerCount
I LargePreviewOffset
I SmallPreviewOffset
I PrintTime
I ProjectorType
f BottomLiftHeight
f BottomLiftSpeed
f LiftHeight
f LiftSpeed
f RetractSpeed
f MaterialMilliliters
f MaterialGrams
f MaterialCost
f BottomLightOffDelay
I Unknown3
H LightPWM
H BottomLightPWM
I LayerXorKey
f BottomLiftHeight2
f BottomLiftSpeed2
f LiftHeight2
f LiftSpeed2
f RetractHeight2
f RetractSpeed2
f RestTimeAfterLift
I MachineNameOffset
I MachineNameSize
B AntiAliasFlag
H Padding
B PerLayerSettings
I ModifiedTimestampMinutes
I AntiAliasLevel
f RestTimeAfterRetract
f RestTimeAfterLift2
I TransitionLayerCount
f BottomRetractSpeed
f BottomRetractSpeed2""".split("\n")


def _aes(data):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    d = Cipher(algorithms.AES(KEY), modes.CBC(IV)).decryptor()
    return d.update(data) + d.finalize()


def settings(b):
    size, off = struct.unpack("<2I", b[4:12])
    s = _aes(b[off:off + size])
    fmt = "<" + "".join(f.split()[0] for f in FIELDS)
    S = {f.split()[1]: v for f, v in zip(FIELDS, struct.unpack_from(fmt, s, 0))}
    mo, ms = S["MachineNameOffset"], S["MachineNameSize"]
    S["MachineName"] = b[mo:mo + ms].decode("utf-8", "replace")
    return S


def _xor_stream(data, seed, idx):
    init = (seed * 0x2d83cdac + 0xd8a83423) & M32
    key = ((idx * 0x1e1530cd + 0xec3d47cd) & M32) * init & M32
    n = len(data)
    keys = (key + np.arange((n + 3) // 4, dtype=np.uint64) * init) & M32
    return (np.frombuffer(data, np.uint8) ^ keys.astype("<u4").view(np.uint8)[:n]).tobytes()


def _runs(rle):
    """Solid runs [start, end) in row-major pixel order."""
    codes, strides, n, L = [], [], 0, len(rle)
    while n < L:
        c, s = rle[n], 1
        if c & 0x80:
            c &= 0x7f
            n += 1
            sl = rle[n]
            if (sl & 0x80) == 0:
                s = sl
            elif (sl & 0xc0) == 0x80:
                s = ((sl & 0x3f) << 8) + rle[n + 1]
                n += 1
            elif (sl & 0xe0) == 0xc0:
                s = ((sl & 0x1f) << 16) + (rle[n + 1] << 8) + rle[n + 2]
                n += 2
            else:
                s = ((sl & 0xf) << 24) + (rle[n + 1] << 16) + (rle[n + 2] << 8) + rle[n + 3]
                n += 3
        codes.append(c)
        strides.append(s)
        n += 1
    st = np.array(strides, np.int64)
    ends = np.cumsum(st)
    on = np.array(codes) > 0x40
    return (ends - st)[on], ends[on]


def layers(b, S, ds):
    """Yield each layer as a downsampled boolean grid (rows = y)."""
    RX, RY = S["ResolutionX"], S["ResolutionY"]
    gx, gy = RX // ds + 1, RY // ds + 1
    for i in range(S["LayerCount"]):
        off = struct.unpack_from("<4I", b, S["LayerPointersOffset"] + 16 * i)[0]
        ld = struct.unpack_from("<I3fIIIIII", b, off)
        data_off, dlen, enc_off, enc_len = ld[4], ld[6], ld[8], ld[9]
        rle = bytearray(b[data_off:data_off + dlen])
        if enc_len:
            rle[enc_off:enc_off + enc_len] = _aes(bytes(rle[enc_off:enc_off + enc_len]))
        if S["LayerXorKey"]:
            rle = bytearray(_xor_stream(bytes(rle), S["LayerXorKey"], i))
        s, e = _runs(bytes(rle))
        grid = np.zeros((gy, gx), bool)
        for a, z in zip(s, e):
            y0, x0 = divmod(int(a), RX)
            y1, x1 = divmod(int(z) - 1, RX)
            for yy in range(y0, y1 + 1):
                xa = x0 if yy == y0 else 0
                xb = x1 if yy == y1 else RX - 1
                grid[yy // ds, xa // ds:xb // ds + 1] = True
        yield i, grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ctb")
    ap.add_argument("--settings", action="store_true", help="print settings only")
    ap.add_argument("--ds", type=int, default=8, help="pixels per check cell")
    ap.add_argument("--json", default=None, help="write islands to this file")
    a = ap.parse_args()
    b = open(a.ctb, "rb").read()
    S = settings(b)
    # (lift, retract and total-height fields don't decode reliably from this
    # block on the Saturn 4 Ultra files seen so far, so they are not shown)
    for k in ("MachineName", "ResolutionX", "ResolutionY", "DisplayWidth", "DisplayHeight", "LayerCount",
              "LayerHeight", "BottomLayerCount", "ExposureTime", "BottomExposureTime",
              "AntiAliasLevel", "MaterialMilliliters", "PrintTime"):
        v = S[k]
        print(f"  {k:22s} {round(v, 4) if isinstance(v, float) else v}")
    if a.settings:
        return
    mmx = S["DisplayWidth"] / S["ResolutionX"] * a.ds
    mmy = S["DisplayHeight"] / S["ResolutionY"] * a.ds
    eight = np.ones((3, 3), bool)
    prev, open_isl, done = None, [], []
    for i, g in layers(b, S, a.ds):
        if prev is not None:
            lab, n = ndimage.label(g, structure=eight)
            if n:
                rest = ndimage.binary_dilation(prev, structure=eight)
                hit = ndimage.maximum(rest, lab, np.arange(1, n + 1))
                for j in np.flatnonzero(hit == 0):
                    ys, xs = np.nonzero(lab == j + 1)
                    open_isl.append(dict(layer=i, x=round(float(xs.mean() * mmx), 2), y=round(float(ys.mean() * mmy), 2),
                                         cells=int(len(xs)), mask=lab == j + 1))
            # follow open islands up until they join something that was
            # already resting (approximation: their region touches a
            # resting piece of this layer)
            if open_isl:
                lab, n = ndimage.label(g, structure=eight)
                still = []
                for it in open_isl:
                    pieces = set(np.unique(lab[it["mask"] & g])) - {0}
                    if not pieces:
                        it["floats"] = i - it["layer"]
                        done.append(it)
                        continue
                    grown = np.isin(lab, list(pieces))
                    if grown.sum() > 4 * it["cells"] + 50 or i - it["layer"] > 200:
                        it["floats"] = i - it["layer"]
                        done.append(it)
                    else:
                        it["mask"] = grown
                        still.append(it)
                open_isl = still
        prev = g
        if i % 200 == 0:
            print(f"  layer {i}/{S['LayerCount']}", flush=True)
    for it in open_isl:
        it["floats"] = S["LayerCount"] - it["layer"]
        done.append(it)
    for it in done:
        it.pop("mask", None)
    done.sort(key=lambda t: -t["floats"])
    lh = S["LayerHeight"]
    print(f"islands: {len(done)}")
    for it in done[:40]:
        print(f"  layer {it['layer']} (z {it['layer'] * lh:.2f} mm) at x={it['x']}, y={it['y']} mm, "
              f"{it['cells']} cells, grows/joins after ~{it['floats']} layers")
    if a.json:
        json.dump(done, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
