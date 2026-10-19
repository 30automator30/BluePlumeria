# 3D Printing with Claude: Lessons Learned

These notes come from building four Blue Plumeria designs in one session:
- the plumeria ring with a twisted vine band
- the heirloom band
- the openwork band
- the fractal lattice plumeria sculpture

Every design was printed on an **Elegoo Saturn 4 Ultra** (resin, 11520 × 5120, 218.88 × 122.88 × 220 mm build volume) and sliced in CHITUBOX. Read this before starting the next design.

---

## 1. Design workflow: what worked

**Start from a reference image, then ask focused questions.** The best results came when Amy sent a reference photo or concept sheet and Claude then asked 3–4 multiple-choice questions: size, use, how it's split into parts, finish, and level of detail. Fuzzy requirements up front cost several redesign rounds later.

**Show it before building it.** Render previews from several angles for approval *before* producing final STLs. Amy asked for this explicitly ("run picture by me for approval first"). Two rounds were thrown away because they went straight to final builds.
- Show a **flat layout** (the design unrolled) for bands. It makes density, gaps and empty spaces obvious.
- Show **side-by-side comparisons** when going back to an earlier version ("see the delta").
- When the user says "the first draft looked better", **restore it as the base** and keep the alternative alongside. Don't keep iterating on the version they rejected.

**Work out the math before the geometry.** For the fractal sculpture, Amy asked to "work out the math first": golden ratio from base to neck, then expanding out. Plotting the proportions (neck at H/φ, radius ÷ φ per quarter turn) before modeling made the design coherent and easy to discuss. Show the user the math chart; it's a design artifact in its own right.

**Follow the reference's structure literally.** When Amy sent a flat-layout concept sheet with an exact order of elements (center flower → leaves → big curls → small curls → side flower → crossing vines → shank), building exactly that order got approved on the first render.

## 2. Code-built (SDF) vs. sculpted (Blender) designs

All designs here were built in Python as **signed distance fields**, meshed with marching cubes into watertight STLs. That approach meant:

**Good at:**
- precise dimensions and ring sizes
- golden-ratio and parametric rules
- repeatable variants: sizes, inner text, stone setting
- guaranteed watertight meshes
- automatic printability checks

**Weak at:** organic, hand-sculpted flow. The first lattice bloom looked "rigid" next to an AI-generated concept image. Flow came from these changes in code:
- **petals:** curved centerlines (a sideways sweep), dip-then-flick tips, one edge rolled higher than the other, and twist that grows toward the tip
- **strokes:** rounded profiles (a "soft band") that swell at the base and taper to the tip
- **lattice:** fan veins plus curving cross-veins instead of a straight-edged honeycomb
- **joins:** soft unions where pieces meet, plus small per-petal variation

**Know the limit:** AI concept images aren't printable as drawn (floating lattice, features too thin to print). Code gets close; matching the exact look needs hand sculpting in Blender or ZBrush. Recommended hybrid: code-generated base mesh, then hand sculpting for character, then a printability check.

## 3. Printability rules (Saturn 4 Ultra, resin)

| Feature | Minimum used | Notes |
|---|---|---|
| Ring wall / band | 1.3–1.7 mm | Comfort-fit inner edge |
| Openwork struts / vines | ~0.6–0.7 mm | Fine for metal casting; use tough resin if wearing the print |
| Lattice struts (sculpture) | 0.9 mm wide | Rims 1.7 mm |
| Engraved lines | ~0.06–0.12 mm deep | Visible at 0.03 mm layers |
| Gaps / voids | ≥ 0.4–0.5 mm | Narrower voids fill in; scripts auto-fill them |
| Inner text | 1.2–1.6 mm cap height, 0.25 mm deep | "ALOHA" engraved inside |

- **Nothing can float.** Every openwork design checks connectivity and refuses to build, or reports, if any piece is detached. Three small flowers once floated free; the check caught it.
- **Hollow bases must drain:** leave them open underneath, or there will be trapped resin and suction cups.
- **Two-part sculptures need a keyed peg** (a flat on the peg stops rotation) and 0.15 mm clearance per side. Verify the parts don't collide with a boolean intersection; it should come out at 0.00 mm³.
- **Ring size:** inner diameter (mm) = 11.63 + 0.8128 × US size. Resin shrinks slightly; print one size first and adjust by ¼–½ size.

## 4. Bugs that cost time (check these first)

1. **Decimating meshes broke watertightness.** Quadric decimation (fast-simplification) created non-manifold edges. Fix: use `manifold3d.Manifold.simplify(tolerance)`, then nudge any coincident vertices apart by 1 µm before exporting the STL, then drop zero-volume specks.
2. **Exact zeros in the SDF** (a flat underside landing exactly on a grid plane) made marching cubes produce duplicate vertices that weren't joined. Fix: `vol[vol == 0] = 1e-6`.
3. **NaN from a fractional power of a negative number:** `np.sin(...) ** 0.7` when sin < 0. Always clip before fractional powers.
4. **Field discontinuities:** initializing the volume to `1e3` while capping evaluated distances lower left seams. Use a consistent cap (3.0 mm) everywhere.
5. **Mesh grid too small:** elements outside the grid bounds were clipped into loose fragments. Size the grid from the actual geometry plus padding.
6. **Ribbons "ribbed" like rope:** the ribbon's orientation flipped at sharp corners where path pieces joined. Fix: carry the orientation smoothly from point to point along the curve, and smooth the path's corners. Protect deliberate tight curls (the ground spirals) from that smoothing.
7. **The stone setting cut through both sides of the ring.** Limit features meant for one side to that side (`z > 0`).
8. **A wrong model name in the docs:** the printer is the **Saturn 4 Ultra**, not the 4 Pro. Confirm hardware details; don't assume.

## 5. Supports: the big lesson

**Use CHITUBOX auto-supports (tree or medium) for final prints.** Claude's job is to deliver **pre-oriented, unsupported STLs** and to check the sliced result.

Writing a support generator in Python worked technically: it ended with zero floating islands after automatic repair passes. But it produced a **support forest**: about 5,500 straight supports, using 96 mL of resin for a 40 mL model, standing in walls around the ribbons. They would be nearly impossible to remove without breaking the lattice. CHITUBOX's branching tree supports put far fewer contacts on the model.

What did work, and is worth keeping:
- **Orientation:** body upright on its flat spiral-root base; bloom face up, tilted 30°, lifted on a raft, so support marks land only on petal backs and the hidden calyx.
- **Island verification by re-slicing:** a layer-by-layer check that catches support tips that miss the model. Two real bugs (tips stopping 0.15 mm short of the surface at both ends) only showed up this way.
- **Repair loop:** add a support at each leftover island and rebuild.
- **Ignore false alarms:** sub-resolution specks and plate-level contact in the first ~0.6 mm.

If the generator gets revisited: branch supports like a tree, and only support true islands plus overhangs steeper than ~60° (not 45°).

## 5b. Designing support-free (the gyroid pendant)

The better answer to supports is a design that doesn't need them. `gyroid-pendant/` prints upright with **0 floating islands** and only a breakaway foot.

- **Gyroids hold themselves up.** A minimal surface has no local low points, so gyroid lace never starts in mid-air.
- **Cutting creates islands, though.** Wherever the flat front or back face slices a gyroid wall that slopes *down toward* the face, the wall's lowest point sits on the face with nothing beneath it (21 of them on the first pendant). Two fixes, in order:
  1. A 0.15 mm erode-then-dilate removes hair-thin slivers.
  2. An automatic repair loop re-slices the part and adds a 0.36 mm, 45° micro-fillet from each remaining island down to the nearest lace below, then repeats.
- **The outline has low points too.** Every downward-pointing lobe of the silhouette is an island. Do a fast 2D scan over rotations to pick the one with the fewest low points (84° left only the two bottom tips), then neck each low point to a single breakaway foot.
- **Use pointed arches for loops.** A round bail has a flat top inside; a teardrop with a 45° roof doesn't.
- **Re-check after every mesh change.** Simplifying to a smaller file brought back one island that a micro-fillet had fixed, so the verification has to run *after* the final simplification.
- **Separate noise from geometry.** Single-pixel "islands" at the checker's resolution are slicing noise. Count only regions of at least 0.01 mm² with nothing within 50 µm below.

## 6. Checking CHITUBOX .ctb files

Claude can read encrypted CHITUBOX `.ctb` files: settings, preview image and every layer. The decryption scheme is the one documented in the open-source UVtools project (AES-CBC settings block, per-layer XOR plus RLE). Workflow:
1. Amy slices in CHITUBOX and sends the `.ctb`.
2. Claude decodes the settings: printer model, layer height, exposure, lift and anti-aliasing.
3. Claude decodes all layers and runs the island check (any region with nothing beneath it).

The first heirloom band file had zero islands across 643 layers. It's a quick, reliable pre-print check.

Writing `.ctb` files from scratch is possible (use a CHITUBOX file as the template) but untested on the printer. Prefer: Claude orients the part → CHITUBOX supports and slices it → Claude checks the `.ctb`.

## 7. Practical / session notes

- **CHITUBOX allows one bound device per account.** "Login failed" with "can bind: 0" means another computer holds the slot: unbind it in the Devices dialog.
- **Large STLs:** files over 100 MB can't go to GitHub. Commit generators, READMEs, previews and reasonable-size STLs; send huge files as zips.
- **Long builds** (full-resolution sculpture parts take 4–5 minutes each; supports take longer) should run in the background with progress checks, not as one blocking command.
- **Point-splat renders** (sampling dense surface points and drawing them with depth) are good enough for design review, without needing a GPU or Blender.

## 8. Where things live

| Design | Folder | Generator |
|---|---|---|
| Twisted vine plumeria ring | `plumeria-ring/` | `plumeria_ring.py` (also shared helpers: `build_mesh`, `decimate`, `render_preview.py`) |
| Heirloom band | `heirloom-band/` | `heirloom_band.py` |
| Openwork band | `openwork-band/` | `openwork_band.py` (approved), `openwork_band_concept.py` (alternative) |
| Self-supporting gyroid pendant | `gyroid-pendant/` | `gyroid_pendant.py` (island repair loop + verification) |
| Fractal lattice sculpture | `fractal-plumeria/` | `lattice_plumeria.py` (bloom + body), `fractal_plumeria.py` (earlier concept), `print_prep.py` (orientation + supports + verification) |
