# Coast suncatcher (stained-glass style)

A round suncatcher, 150 mm across, printed in clear resin and then painted by hand with glass paint. Raised "lead" lines on a clear pane close off every piece, so each one holds its own colour like a little dam.

The scene is original, in a hand-cut stained-glass style:
- **Cliff:** a cliff on the left, cut into long vertical strips, with jungle on top.
- **Surf:** white water crashing at the cliff's foot and rolling toward the viewer.
- **Sun:** setting on the horizon, with rays fanning across the whole sky.
- **Headland:** a paler headland behind the cliff, for depth.
- **Sea:** long streaks, with the sun's path on the water.
- **Flowers:** red and orange hibiscus and four plumerias (two white, two pink), with leaves and rocks, along the bottom.

| File | What it is |
|---|---|
| `suncatcher.py` | The generator. `--voxel 0.2 --preview` gives a quick look. `--voxel 0.1` is the final build, and `--guide-only` makes just the paint guide. |
| `wave_suncatcher.stl.zip` | The final STL, zipped. Unsupported and not oriented: lay it out as below. |
| `paint_guide.png` | Left: the painted mock-up with suggested colours. Right: the lead lines as printed, for planning colours. |
| `preview_stained.png` | Renders of the clear print. |

## Design
- **Size:** 150 mm across, or 162.6 mm tall with the hanging loop, and 4.4 mm thick.
- **Pane:** clear, 1.6 mm thick, flat on the back.
- **Lead lines:** 1.0 mm wide and 1.0 mm above the pane, with rounded tops and straight sides so paint stays in its piece.
- **Frame:** a rounded ring 5 mm wide and 4 mm thick.
- **Loop:** a 4 mm hanging hole at the top.
- **Pieces:** about 255. Any piece smaller than 5 mm² or thinner than about 2.3 mm is merged into a neighbour, so every piece can be reached with a fine brush or a paint bottle with a fine tip.
- **How it's drawn:** the scene is a map of labelled regions. Big areas (sky, sea, rocks, surf) are cut into irregular pieces by a Voronoi pattern measured in a gently warped, stretched space, so the cuts bow like hand-cut glass and lie along the way each thing flows. The rays, cliff strips, flowers and leaves are drawn directly. Lead lines run along every boundary, plus a few free lines: the hibiscus veins and stamens.
- **Resolution:** the design is always drawn at 0.2 mm (`DESIGN_RES`), because the cut pattern depends on the pitch. The final mesh resamples the smoothed line field onto a 0.1 mm grid, so the final matches the approved preview.

## Printing
- **Orientation:** stand it up with the loop on top. Lean it 30–45° from vertical so that, in CHITUBOX, the painted face looks up and the back faces the build plate.
- **Supports:** put CHITUBOX tree supports on the frame's lower edge and the flat back only. Keep them off the painted face, and never put any inside the pieces.
- **Finish:** wash and cure well. Clear resin yellows if it's over-cured, so keep the post-cure short. A thin clear coat on the back helps it look like glass.
- **Paint:** glass paints (for example Gallery Glass or Pebeo Vitrea 160) flowed into each piece. The lines are 1 mm tall, so fill each piece to just below the top of its lines.

## Final build
- **Mesh:** 0.1 mm voxels, then decimated with `manifold3d`. 1.44M faces, watertight after an STL save and reload, one body, consistent winding.
- **Size and resin:** 38.1 mL, 150.0 × 162.6 × 4.4 mm.
- **Matches the preview:** the cut pattern is the approved one (the paint guide is pixel-identical) and has 253 pieces.
- **Slice check:** stood up with the loop on top, at 0°, 30° and 45° lean, every layer touches the one below, so there are no islands. The only start point is the bottom of the frame, which the supports hold.
- **File:** the STL is 72 MB, zipped to 29 MB.
