# Hibiscus bonsai

A braided hibiscus bonsai, about 120 mm tall in its pot. It has two parts, and both print straight on the plate with no supports.

| File | What it is |
|---|---|
| `bonsai_hibiscus.py` | The generator: tree and pot. `--preview` skips support planning. |
| `final_stems.pkl` | The 26 print stems in the approved final build. These are the planner's stems plus repair stems added after the full-resolution check. |
| `preview.png` | Renders of the final build. |

## Design
- **Trunk (`TRUNK_STYLE = "braid"`):** four smooth stems woven over and under like a braided hibiscus standard, with two winding each way and one turn per 42 mm. The trunk tapers about 40% to the crown.
- **Base:** near the soil the stems spread into a flared base, standing out as buttresses, and run on as roots that dive into the soil. Four smaller roots run between them.
- **Rope style (`"rope"`):** the earlier version. Four plaited cords spiral up from golden curls on the soil.
- **Crown:** the four stems part into four branches spaced evenly, each forking into two twigs.
- **Middle tier:** plaited arms roll into question-mark curls, with leaves.
- **Flowers:** two open hibiscus, about 40 mm across, on opposite sides of the crown. They have solid ruffled petals with raised veins and an eye ridge, plus an anther column and stigma pads. A twisted bud sits on another twig.
- **Leaves:** 53 serrated hibiscus leaves. A leaf that would pass through a flower is swung clear, or dropped if it can't be.
- **Pot:** a tapered rectangle with a lip, corner feet and two drain holes. The soil plug rests on a 45-degree ledge with 0.3 mm clearance.

## Printability
- **Supports:** none needed. Leaf stalks run straight into their twigs. Petal ruffles stay in the outline and fade out at the tips, because height ruffles made dips. Flowers lean only 8 degrees.
- **Print stems:** anything still starting in mid-air gets a slim stem. A stem may only take a path that is clear of other parts.
- **Final check:** at 0.05 mm layers, the tree took repair rounds from 3 islands to 2, 1, then 0. The lighter copy (0.015 mm simplification, about 0.9M faces) and the pot also check clean.
- **Memory:** a full 0.12 mm build uses about 14 GB. Run each repair round in its own process.
