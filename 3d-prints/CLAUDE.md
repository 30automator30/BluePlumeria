# 3d-prints: instructions for Claude

Resin jewelry and sculpture for Blue Plumeria, printed on an **Elegoo Saturn 4 Ultra** and sliced in **CHITUBOX**. Read `LESSONS_LEARNED.md` before starting new work. The key rules:

- **Get approval on preview renders first.** Never build final STLs before Amy approves the look. Show flat layouts for bands and side-by-side comparisons for revisions.
- **Ask focused questions** about size, use, part split and finish when given a reference image. Follow concept sheets' element order literally.
- **Every part must be watertight and connected.** Run the connectivity check, and use `manifold3d` simplify for decimation, never fast-simplification.
- **Minimums:** struts ≥ 0.6 mm (0.9 mm for sculpture lattice), voids ≥ 0.4 mm, open-bottom hollow parts, keyed pegs with 0.15 mm clearance.
- **Prefer self-supporting designs** (see `gyroid-pendant/`). Re-slice and check for islands *after* the final mesh simplification, because simplifying can bring islands back.
- **Supports:** deliver pre-oriented, unsupported STLs and let CHITUBOX tree supports do the rest. Offer to check the sliced `.ctb` for floating islands.
- **Files:** keep any single file under 100 MB in git; zip and send larger STLs instead.

## Workflow for every new piece

1. **Ask.** Use focused multiple-choice questions: subject, size, use, part split, finish and detail. Follow a reference's structure literally, but make an original design, never a copy of someone else's product.
2. **Preview.** Run the generator's `--preview` or quick mode, then `tools/render_views.py` (with `--up y` for flat pieces). Send the renders, plus a painted mock-up or flat layout where one helps. Revise until Amy approves, following her markup literally. If a direction is rejected twice, change approach; don't keep tuning it.
3. **Final build.** Only after approval. Run the generator's final mode or `final_build.py`, then check that the result is watertight, one body, and has no islands after the final simplification (`tools/slice_check.py` for an STL, the design's own island loop for sculptures).
4. **Deliver.**
   - Zip the STL if it's over 100 MB, or if it's large to send.
   - Send it with orientation and support notes.
   - Write the design's `README.md` (files, design, printing, final-build numbers).
   - Add a `LESSONS_LEARNED.md` entry and a row in its section 8 table.
   - Commit and push. Every script the build needs must be in the repo, never only in a session's scratch folder.
5. **After slicing.** Offer to run `tools/ctb_check.py` on Amy's `.ctb`.

## Shared code

- `tools/`: `render_views.py` (previews), `slice_check.py` (STL islands), `ctb_check.py` (CHITUBOX files). See `tools/README.md`.
- `plumeria-ring/plumeria_ring.py`: `decimate` (manifold3d) and `smin`.
- `fractal-plumeria/fractal_plumeria.py`: `mesh_volume` (marching cubes).
- `fractal-plumeria/lattice_plumeria.py`: `find_islands` and the stem planner.
- `stained-glass/stained_glass.py`: the suncatcher engine. To make a new stained-glass suncatcher, follow `stained-glass/README.md`:
  - Write a scene file.
  - Iterate on the paint guide with `--guide-only`.
  - Use hand-cut pieces, never grids or spirals.
  - Build the final at `--voxel 0.1` with the fixed 0.2 mm design pitch.
