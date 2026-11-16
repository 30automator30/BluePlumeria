# Fractal lattice plumeria sculpture

A 158–160 mm sculpture designed with golden-ratio proportions. A lattice plumeria bloom sits on a body of twisting ribbons that rise as golden spirals, pinch at a neck, and fan out to a second bloom and small flowers. The bloom fits the body on a keyed peg.

| File | What it is |
|---|---|
| `lattice_plumeria.py` | The current generator, for the bloom and body: `--part bloom\|body\|both --voxel 0.12 --outdir DIR`. It also holds shared helpers other designs use: `find_islands` (layer-by-layer island finder on an SDF volume) and the stem planner (`plan_stems`, `stems_for`, `curls_for`), which grows S-curved stems and question-mark curls up to anything that would start in mid-air. |
| `fractal_plumeria.py` | The earlier concept, kept because other designs import its `mesh_volume` (marching cubes, keeping the largest piece) and `capsule_chain`. |
| `print_prep.py` | An experimental orientation and support generator. It over-supports, so don't use it for real prints; let CHITUBOX tree supports do it. |

## Notes
- The design came from Amy's concept sheet and a golden-ratio math chart: neck at H/φ, radius ÷ φ per quarter turn. Showing the math chart first made the design easy to discuss (lessons §1).
- The body prints straight on the plate with no supports. The base is cut flat, and stems and curls are planned in until nothing starts in mid-air (lessons §5d).
- Supports broke the delicate lattice on the first print. See lessons §5c for CHITUBOX settings for delicate parts.
- Builds take several minutes per part at full resolution, so run them in the background.
