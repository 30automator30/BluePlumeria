# ʻŌhiʻa lehua bonsai

An ʻōhiʻa lehua bonsai with an ʻiʻiwi, about 120 mm tall in its pot. It has two parts, and both print straight on the plate with no supports.

| File | What it is |
|---|---|
| `ohia_bonsai.py` | The generator: tree and pot. `--preview` skips support planning. |
| `final_build.py` | The final build: the tree at 0.12 mm with the blossoms and bird re-meshed at 0.06 mm, a printer-resolution island check and repair rounds, a lighter copy, and the pot. |
| `final_stems.pkl` | The 2 print stems in the approved final build, added by the printer-resolution repair round. |
| `preview.png` | Renders of the final build. |

## Design
- **Trunk:** a gnarled informal upright. Five stems fuse into one trunk, twisting half a turn so the bark reads as deep twisted ridges. They flare into buttress roots that dive into the soil.
- **Base:** a surface root crawls over a lump of pitted lava rock, and an aerial root drops from the long low branch into the soil.
- **Branches:** four branches and a leader, each ending in a lehua blossom over a cushion of leafy sprigs.
- **Blossoms:** each is a cluster of 11 flowers and a few closed buds on a knob. Every flower is a stalk swelling into a rounded cup, with a fan of 12 long stamens tipped with bead-like anthers.
- **Foliage:** small rounded leaves (widest beyond the middle), in opposite pairs turned 90 degrees at each node, crowding into rosettes at the sprig tips. Sprigs end in a red leaf bud (liko) or a popcorn cluster of round flower buds.
- **ʻIʻiwi:** perched on the right branch at 1.3× scale (`BIRD_S`, about 33 mm from bill tip to tail). Its down-curved bill dips into one of the blossom's lower flower cups. It has a round head with bead eyes, folded wings in layered feather rows, and a notched tail resting on the branch. Leaves and stamens are kept clear of it.
- **Pot:** an oval with a lip, four feet and two drain holes. The soil plug rests on a 45-degree ledge with 0.3 mm clearance.

## Printability
- **Stamens:** every stamen leaves its flower at least 25 degrees above horizontal and curls upward. Each anther swells gradually from the filament, so its tip is always the highest point.
- **Flowers and buds:** these point at least 40 degrees up, so the cups, which widen at about 30 degrees, rise from their stalks.
- **Leaves and buds on sprigs:** leaf stalks end inside their twig, and bud clusters rise on a neck from the sprig tip. Both used to poke out underneath and start in mid-air.
- **Bird:** its feet stay on top of the branch (side toes hung off it at first). Its belly feathers rest on the branch under its lowest point. Its tail tip rests on the branch, and its bill tip sits in a flower cup.
- **Stamen thickness:** filaments are about 0.7–0.8 mm thick with 1.1 mm anthers. Handle the blossoms gently when washing and removing supports from the plate.

## Final build
- **Tree:** 8.9M faces, watertight (and still watertight after an STL save and reload), 26.7 mL of resin, 97 × 84 × 111 mm. A 0.05 mm-layer slice check finds 0 islands, after 1 repair round added 2 slim stems.
- **Lite copy:** 2.1M faces (0.012 mm simplification). It also checks at 0 islands and reloads watertight.
- **Pot:** 47k faces, watertight, 0 islands.
