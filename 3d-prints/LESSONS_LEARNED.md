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
- **Ring size:** inner diameter (mm) = 11.63 + 0.8128 × US size, **plus a 0.5 mm fit allowance**. Test prints in Elegoo standard resin came out small (a US 8 measured about 7.25–7.5), so the generators add `FIT_ALLOWANCE = 0.5` mm (`--fit`). For silver casting use about 0.8 mm.

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

## 5c. Support removal broke the first prints

Every first-round print broke while supports were removed. The support contacts (0.5 mm) were nearly as thick as the vines (0.6–0.7 mm), and standard resin is brittle once fully cured. Fixes:
- **Contact size:** contact diameter ≤ ⅓ of the feature it touches (about 0.25 mm).
- **Support placement:** only on sturdy areas.
- **Removal:** remove supports before the final cure, on a warmed print, with flush cutters; never snap.
- **Resin:** tough/ABS-like for wearing; castable for metal. Each generator's `--metal` option builds thicker-wire, deeper-engraving versions for casting and plating.

## 5d. Designing the supports into the sculpture

- Amy's direction: turn supports into design. The sculpture body now prints straight on the plate with **no supports**. It has a flat-cut base, short roots, two question-mark curls (her sketch) under the second bloom and the left flower, and slim S-curved stems that branch off the curls to the remaining low points.
- Stems leave and arrive vertically and never lean past ~50°, so they print without supports themselves.
- Plan stems on a coarse grid (0.3 mm, fast), but **always re-check the final mesh at printer resolution** (`verify`, 0.05 mm layers). The coarse plan missed 9 islands of 0.01–0.3 mm². The repair loop (find islands, add a stem from the nearest solid below, rebuild) took two rounds.
- The first layer on the plate is not an island; skip it in the check.
- Simplifying the 3.4M-face body at 0.02 mm brought back two 0.01 mm² specks; 0.012 mm (1.8M faces) stayed clean. Re-verify every lighter copy.
- Petal lattice: veins must run *into* the next frame. Veins that stop at a gap leave loose stubs, and those are what snapped during support removal. Bolder rims (2.4 × 2.3 mm) and struts (1.25 × 1.5 mm) survive removal.

## 5e. Hibiscus bonsai: printed clean, flat on the plate, no supports

- The approach worked first time: both parts printed flat on the plate with no supports. Leaves rise from their stalks, petal ruffles stay in the outline, and a few slim stems catch anything left over.
- CHITUBOX auto-orient tilts these parts and adds a forest of supports. Undo it: set rotation back to 0, place on bed, lift 0, delete supports.
- Fixes that removed most islands at the source: leaf stalks must run into the twig (a stalk dipping below its twig floats); a scalloped outline instead of height ruffles; flowers lean no more than about 8 degrees; anther beads seated in the column.
- Check for clipping (a flower through a flower, leaves through petals) before building. Swing leaves clear of flowers; route stems only through clear paths.
- Trunk: a smooth four-stem braid with a flared base (stems running on as roots into the soil) read as a tree. A rope of plaited cords looked busy and leggy.
- A full 0.12 mm build uses about 14 GB, so run each repair round in a fresh process.
- Cleanup: white flecks on edges are dried resin or wash haze; re-brush with fresh IPA, dry, then a short re-cure.

## 5f. Tropical arrangement (bird of paradise + pride of Barbados): printed clean

- Fine detail survived in standard resin: 0.7 mm stamens with anther tips, stigma whiskers, crest channels, colour-band guide grooves of about 0.13 mm, and leaf splits.
- Leaves must cup up and inward. Curling them outward and down made their edges and tips hang, which needed 97 stems; cupping them cut that to 12.
- Flowing S-sway on stalks: fade it out at both ends (sin(pi s)^1.5), or the leaf it carries changes angle and starts hanging.
- Pointy add-ons (arrowhead barbs, beads on whiskers) start in mid-air. Widen barbs gradually up to the notch, and taper whiskers instead of capping them with a bead.
- Multi-resolution build: re-mesh the heads at 0.06 mm in their own boxes and join them to the 0.12 mm body. Keep every piece inside each box, because leaves crossing it are separate pieces there; dropping them cut leaves off flat at the box walls. Always render the final file before sending.
- Container restarts cost hours on long builds. Save the stems and the STL each round, and resume from them.

## 5g. ʻŌhiʻa lehua bonsai with ʻiʻiwi: final build notes (not yet printed)

- Lehua stamens: every stamen leaves its flower at least 25 degrees up and curls upward. Anthers swell gradually from the filament (club, not bead), so the tip is the highest point. Flower cups widen at about 30 degrees, so flowers must point at least 40 degrees up.
- Leaf stalks must end inside their twig. A petiole that ran through a thin sprig poked out underneath and started in mid-air. The same goes for anything hung under a sprig tip, such as bud-cluster necks: start them at the tip.
- A perched bird: keep the feet on top of the branch (within about 0.7 × branch radius sideways), since toes wrapping the sides hang in mid-air. A pitched body's lowest point sits behind the legs, so rest the belly feathers on the branch there. Rest the tail tip on the branch and put the bill tip into a flower.
- Check islands at 0.2 mm before the final build. It found 18 design-level islands in under a minute. The 0.05 mm final check then needed only 2 stems.
- STL watertightness: Manifold can leave distinct vertices that coincide at float32 precision, and STL welds them on reload, giving a few non-manifold edges. Find twins at float32, not by rounding to 1e-6. Push each twin a different amount (twins at a pinch can share a normal), and check a simulated STL weld before saving. Keep the exact topology (npz) for later steps such as the lite copy.

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
| ʻŌhiʻa lehua bonsai with ʻiʻiwi | `ohia-bonsai/` | `ohia_bonsai.py` (tree + pot), `final_build.py` (fine blossoms and bird, island repair, lite copy, STL-weld check) |
