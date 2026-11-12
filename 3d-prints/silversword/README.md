# ʻĀhinahina (Haleakalā silversword)

An ʻāhinahina in bloom with a nēnē, about 153 mm tall on an oval cinder plinth. It is one part and prints straight on the plate with no supports.

| File | What it is |
|---|---|
| `silversword.py` | The generator. `--voxel 0.25 --preview` gives a quick look. |
| `final_build.py` | The final build: the plinth and core at 0.15 mm, with the spike, rosettes and nēnē re-meshed at 0.06 mm in 48 tiles (each built in its own process and cached), a printer-resolution island check, repair rounds and a lighter copy. `--lite` remakes only the lighter copy. |
| `patch_islands.py` | Pins islands found just above the plinth with short posts into the cinders, without rebuilding the tiles, and re-checks each spot on a local crop. |
| `final_stems.pkl` | Print stems in the final build: none (the design checked clean). |
| `final_pins.json` | The 2 pins `patch_islands.py` added in the final build. |
| `preview.png` | Renders of the final build. |

## Design
- **Spike:** a tall egg-shaped cob of 166 daisy-like flower heads in a golden spiral (`HEAD_GAP` 6.1 mm). Each head is a cup on a short stalk, a domed centre pitted with florets and a ring of tubular ray petals. Heads shrink toward both ends, with buds at the tip and slim bracts at the foot.
- **Rosette:** a ball of silver needle leaves. The upper needles rise and curve upward, and a skirt of lower needles bows out and droops until the tips rest on the cinders.
- **Small rosettes:** three young rosettes around the plinth, at 0.42, 0.36 and 0.3 scale.
- **Nēnē:** a Hawaiian goose resting on one leg, looking up at the bloom. It has a furrowed neck, a black head with a short bill and bead eyes, folded wings in four layered tiers with barred edges, a short tail and a webbed foot.
- **Plinth:** a 124 × 84 mm oval, 3 mm at the rim, with an irregular pebbled cinder surface and half-sunk cinders.

## Printability
- **Heads:** every head faces at least 40° up, the cups widen at no more than 35°, and each ray petal is cupped forward to point at least 8° up. Floret texture is cut in as pits, never raised.
- **Cups:** these are true cones, not chains of capsules, and the domes keep only their front half. Cups, domes and rays are joined with a hard union (no blending), and the ray roots are buried in the cup. Each of these fixes removed islands where something bulged below the cup.
- **Needles:** they rise, or droop all the way to the ground. Skirt tips stay inside the plinth outline.
- **Nēnē:** its standing leg is under the body's lowest point, and the rear wing tiers ride up onto the back.

## Final build
- **Full:** 8.24M faces, watertight (and still watertight after an STL save and reload), 76.0 mL of resin, 126.5 × 86.4 × 152.8 mm. A 0.05 mm-layer slice check found 2 tiny islands. Both were skirt-needle tips hovering just above the cinders, and each got a 0.7 mm pin into the plinth (`final_pins.json`). Both spots re-check clean.
- The 0.06 mm tiles total 32.7M faces before joining, so each tile is simplified to 0.012 mm as it is joined, which is under the printer's ~19 µm pixel.
