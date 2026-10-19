# 3d-prints: instructions for Claude

Resin jewelry and sculpture for Blue Plumeria, printed on an **Elegoo Saturn 4 Ultra** and sliced in **CHITUBOX**. Read `LESSONS_LEARNED.md` before starting new work. The key rules:

- **Get approval on preview renders first.** Never build final STLs before Amy approves the look. Show flat layouts for bands and side-by-side comparisons for revisions.
- **Ask focused questions** about size, use, part split and finish when given a reference image. Follow concept sheets' element order literally.
- **Every part must be watertight and connected.** Run the connectivity check, and use `manifold3d` simplify for decimation, never fast-simplification.
- **Minimums:** struts ≥ 0.6 mm (0.9 mm for sculpture lattice), voids ≥ 0.4 mm, open-bottom hollow parts, keyed pegs with 0.15 mm clearance.
- **Prefer self-supporting designs** (see `gyroid-pendant/`). Re-slice and check for islands *after* the final mesh simplification, because simplifying can bring islands back.
- **Supports:** deliver pre-oriented, unsupported STLs and let CHITUBOX tree supports do the rest. Offer to check the sliced `.ctb` for floating islands.
- **Files:** keep any single file under 100 MB in git; zip and send larger STLs instead.
