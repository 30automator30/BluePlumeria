# Tropical arrangement: bird of paradise with pride of Barbados

A desk sculpture about 82 × 81 × 125 mm and 30.6 mL of resin. It prints in one piece upright on its plinth with no supports.

| File | What it is |
|---|---|
| `bird_of_paradise.py` | The generator. `--preview` skips support planning. |
| `final_build.py` | The final build. Each head is re-meshed at 0.06 mm and joined to the 0.12 mm body. The island check repeats with extra stems until it comes back clean, then a lighter copy is made and checked. |
| `final_stems.pkl` | The 45 print stems in the approved final build. |
| `preview.png` | Renders of the final build. |

## Design
- **Plinth:** two stepped squares with chamfered edges, sized by the golden ratio.
- **Clump:** leaf sheaths twist round each other at the base. Seven split paddle leaves on swaying S-curved stalks curve up and inward and twist.
- **Curls:** four golden-spiral tendrils curl out from the base, as on the plumeria.
- **Blooms:** three large bird of paradise heads at staggered heights. Each has:
  - a sheathing bract where the beak meets the stalk;
  - a sharp-keeled, upturned beak with colour-band guide lines;
  - three crest sepals with a keel, a channel and drawn-out tips, plus pale throat bracts;
  - the blue arrowhead tongue with barbs and a stigma whisker;
  - a younger crest and tongue behind.
- **Pride of Barbados:** a spike of five frilly, cupped flowers. Each has a raised standard petal and a fan of stamens; round buds sit at the tip.

## Printability
- **Leaves:** each leaf cups up and inward with a tapered base. Leaves that curled outward and down hung their edges and tips, which needed 97 stems.
- **Tongue barbs:** they widen gradually up to the notch. Abrupt lobes hung.
- **Stigma whisker:** it tapers to a rounded tip; a bead on the end started as a speck in mid-air.
- **Fine heads:** keep every piece inside the head's box. Leaves crossing the box are separate pieces there, and dropping them cut them off flat at the box walls.
- **Final check:** at 0.05 mm layers, repair rounds took the island count from 1 to 0. The lighter copy (about 1.0M faces) also checks clean.
