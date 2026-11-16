# Stained-glass suncatcher engine

`stained_glass.py` builds stained-glass-style suncatchers. They print in clear resin and Amy paints them by hand with glass paint. Each design is a short scene file that draws a picture; the engine does the rest:
- splits the picture into pieces and merges any too small to paint;
- runs raised lead lines along every boundary;
- adds the pane, frame ring and hanging loop;
- draws the paint guide;
- builds the final STL.

The first design is `../wave-suncatcher/suncatcher.py` (a coast scene). Copy it to start a new one.

## Making a new suncatcher
1. **Ask first.** Get the subject, size (150 mm is typical; the build plate is 218 × 123 mm), level of detail (bold about 50 pieces, medium about 100, fine 200–400), and which reference to follow. Amy's references are hand-cut stained glass.
2. **New folder and scene file.** Copy `wave-suncatcher/suncatcher.py` and keep its `FRAME`, `OUT`, `PALETTE` and `draw(M)` shape. Inside `draw`:
   - `M.paint(mask, cat)` paints one piece, and `M.paint_split(mask, keys, cat)` paints one piece per key. Later paint goes on top.
   - `sg.shatter(M, mask, cat, spacing, ang, stretch, amp, seed, tf)` cuts a region into irregular hand-cut pieces. Long pieces lie along `ang`; `tf` can add perspective. `cat` can be a function of the piece centre, which is how colour gradients work.
   - `sg.hibiscus`, `sg.plumeria` and `sg.leaf` give ready-made subjects.
   - `M.line(dist)` adds free lines, such as veins and stamens, that may end inside a piece.
   - `sg.poly_sdf`, `sg.blob_poly`, `sg.petal_poly`, `sg.hib_petal` and `sg.leaf_poly` give shapes.
3. **Iterate on the paint guide.** `python3 scene.py --guide-only` takes about a minute and draws the painted mock-up beside the line drawing. That's what Amy judges; the clear-print render barely shows the design. Send her both, and follow her markup literally.
4. **Preview mesh.** `python3 scene.py --voxel 0.2 --preview`, then `../tools/render_views.py OUT.png part.stl --up y --close=X,Y,Z`.
5. **Final, after approval.**
   - Run `python3 scene.py --voxel 0.1`. The cut pattern is always drawn at `FRAME.design_res` (0.2 mm), so the final keeps the approved pattern. Check that the paint guide is pixel-identical to the approved one.
   - Run `../tools/slice_check.py part.stl --stand --lean 0 30 45 --layer 0.1 --xy 0.1`.
   - Zip the STL (a 150 mm piece is about 70 MB).
   - Write the README and add a lessons entry.

## What Amy liked and didn't
- **Rejected:** geometric cuts, meaning grids, even radial sectors, straight bands and spirals. They read as computer-made.
- **Approved:** irregular warped pieces that lie along each thing's flow, rays fanning across the whole sky, long vertical strips with a few slanted cuts for cliffs, and realistic overlapping petals.
- **Too busy:** many tiny jagged facets read as noise.

## Defaults (`Frame`)
- Pane 1.6 mm with a flat back.
- Lines 1.0 mm wide and 1.0 mm tall, with rounded tops and straight sides.
- Frame ring 5 mm wide and 4 mm thick.
- Loop hole 4 mm.
- Pieces under 5 mm² or under about 2.3 mm wide are merged.

## Printing
- Stand it up with the loop on top. Lean it 30–45° so that, in CHITUBOX, the painted face looks up and the back faces the build plate.
- Put tree supports on the back and the frame's lower edge only.
- Standing straight up leaves an overhang ledge of about 37 mm² at the loop and frame top.
