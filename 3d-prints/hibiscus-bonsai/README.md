# Hibiscus bonsai

A braided hibiscus bonsai, about 120 mm tall in its pot. It has two parts, and both print straight on the plate with no supports.

| File | What it is |
|---|---|
| `bonsai_hibiscus.py` | The generator: tree and pot. `--preview` skips support planning. |
| `final_stems.pkl` | The 35 print stems in the approved final build. These are the planner's stems plus repair stems added after the full-resolution check. |
| `preview.png` | Renders of the final build. |

## Design
- **Trunk:** four cords, each a three-strand plait. They rise from flat golden curls on the soil plug and spiral in as a golden vortex, like the plumeria body's base. They meet at the golden height and twist together, four into one, up to the crown.
- **Crown:** the four cords part into four branches, each forking into two twigs.
- **Middle tier:** sweeps out both ways as plaits that roll into question-mark curls, with leaves.
- **Flowers:** two open hibiscus, about 40 mm across, on opposite sides of the crown. They have solid ruffled petals with raised veins and an eye ridge, plus an anther column and stigma pads. A twisted bud sits on another twig.
- **Leaves:** 51 serrated hibiscus leaves. Any leaf that would pass through a flower is dropped.
- **Pot:** a tapered rectangle with a lip, corner feet and two drain holes. The soil plug rests on a 45-degree ledge with 0.3 mm clearance.

## Printability
- **Supports:** none needed. Leaf stalks run straight into their twigs. Petal ruffles stay in the outline and fade out at the tips, because height ruffles made dips. Flowers lean only 8 degrees.
- **Print stems:** anything still starting in mid-air gets a slim stem. A stem may only take a path that is clear of other parts.
- **Final check:** at 0.05 mm layers, the tree took three repair rounds (13, then 2, then 1, then 0 islands). The lighter copy (0.015 mm simplification, about 1.08M faces) and the pot also check clean.
- **Memory:** a full 0.12 mm build uses about 14 GB. Run each repair round in its own process.
