# Honu over coral

A green sea turtle gliding over a coral head on an oval sand base, about 130 mm long. It is one part. The turtle is held up by two flippers: the left front flipper rests in a fork of finger coral, and the left rear flipper trails onto a coral pinnacle.

| File | What it is |
|---|---|
| `honu.py` | The generator. `--voxel 0.3` gives a quick look. |
| `final_build.py` | The final build: the body at 0.2 mm, re-meshed at 0.08 mm in 56 tiles (each built in its own process and cached, with empty tiles skipped), a printer-resolution island check and a lighter copy. `--round` runs one build and check, and `--lite` remakes only the lighter copy. |
| `patch_islands.py` | Pins islands the final check finds, without rebuilding the tiles. Each post grows downward until a local crop re-checks clean. |
| `final_stems.pkl` | Print stems in the final build: none. |
| `final_pins.json` | The 3 posts `patch_islands.py` added in the final build, as `[z, x, y, length]`. |
| `preview.png` | Renders of the final build. |
| `preview_concept.png` | The approved concept render. |

## Design
- **Honu:** a green sea turtle shell with raised scutes, a neck and head with eyes, scaled flippers and a short tail. The shell is hollow (2 mm wall) and drains through two 1.2 mm holes in the plastron.
- **Coral:** a lobe-coral mound with a knobbly pinnacle under the rear flipper and a small knoll. The finger coral is knobbly and tapering, and each finger splits into 2–3 rising tips. There are two cauliflower clumps and a small finger clump at the back. Corallite pits are cut into all the coral.
- **Sand:** a 128 × 78 mm oval with ripples, holding an urchin, a cowrie, 10 seagrass clumps and 6 halimeda limu fans made of flat, wider-than-tall segments.
- **Revision:** in the first final build, the fingers had round caps, the flipper rested on a round knob, and the limu looked like stacked beads. Amy found these too phallic, so they were reworked as described above.

## Printability
- **Base:** the sand base is solid (2.6 mm at the rim). Only the coral mound is hollow, with two pitched tunnels to the rim so it drains.
- **Coral:** branches are joined with a plain union (no blends, which web between fingers), and every coral tip keeps rising at 40° or more.
- **Cleanup:** voids, specks and crumbs are cleaned up in the volume before meshing.
- **Supports:** the turtle's belly and the undersides of the flippers are broad, gentle overhangs. They grow from connected areas, so they aren't islands, but they're too flat to print safely without supports. Let CHITUBOX add light tree supports under them.

## Final build
- **Full:** 3.36M faces, watertight (and still watertight after an STL save and reload), 0 STL weld faults, 63.2 mL of resin, 130.2 × 102.6 × 81.0 mm.
  - A 0.05 mm-layer slice check found 4 tiny islands. Three got short posts (0.9, 0.9 and 1.2 mm, each including 0.6 mm buried in the solid below; listed in `final_pins.json`), and the fourth checked clean on its local crop.
- **Lite copy:** being made.
