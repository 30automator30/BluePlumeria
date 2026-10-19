# Handoff prompt: continue the Blue Plumeria 3D-print work

Paste everything below the line into a new Claude Code session. Running it locally on Amy's PC is best: that session can also read `I:\Lost Pines Creative\BluePlumeria\3D prints`.

---

You're continuing 3D-print design work for **Blue Plumeria** (Amy's jewelry business). Everything lives in the `BluePlumeria` repo under `3d-prints/`, on branch `ccr-3ea93a43-bwsqz4`. Amy's local working copy is `I:\Lost Pines Creative\BluePlumeria`; sliced CHITUBOX files are in `I:\Lost Pines Creative\BluePlumeria\3D prints`.

**Read these first:**
- `3d-prints/CLAUDE.md`: the rules (approval renders before final builds, watertight and connected parts, printability minimums, supports, file sizes).
- `3d-prints/LESSONS_LEARNED.md`: everything learned so far: design workflow, the bugs that cost the most time, supports, support-free gyroid design, and checking `.ctb` files.

**Equipment:** Elegoo **Saturn 4 Ultra** (resin, 218.88 × 122.88 × 220 mm build volume), sliced in **CHITUBOX**. CHITUBOX allows one bound device per account; if login fails, unbind the old device in the Devices dialog. Amy currently uses Elegoo **standard** resin and plans to get tough/ABS-like or castable resin.

**Designs (each folder has a Python generator, a README, previews and STLs):**
| Folder | Design |
|---|---|
| `plumeria-ring/` | Plumeria ring with a twisted three-vine band and edge rails. `plumeria_ring.py` also holds the shared helpers: `build_mesh`, `decimate`, `us_size_to_inner_diameter`, `FIT_ALLOWANCE`. |
| `heirloom-band/` | Hawaiian heirloom engraved band with scalloped edges |
| `openwork-band/` | Pierced openwork plumeria band, 30 mm front section |
| `gyroid-pendant/` | Self-supporting gyroid plumeria pendant. v2 is current (pointed pinwheel petals, 2.6 mm thick); it prints with no supports and verifies with 0 islands. |
| `fractal-plumeria/` | 158 mm lattice plumeria sculpture: `lattice_plumeria.py` (bloom + body); `print_prep.py` is an experimental support generator that over-supports, so don't use it for real prints |

**Current state and decisions:**
1. **Ring sizes ran small.** A printed US 8 measured about 7.25–7.5, so all ring generators now add `FIT_ALLOWANCE = 0.5` mm to the inner diameter (`--fit` changes it). For casting in silver, use about `--fit 0.8` to also cover metal shrinkage. Confirm the fit with a test print before batch printing.
2. **Supports broke the delicate parts** (0.5 mm support contacts on 0.6–0.7 mm vines). Recommended CHITUBOX "Top" settings for delicate parts:
   - contact diameter 0.25 mm (0.30 for the sculpture)
   - contact depth 0.20 mm
   - upper diameter 0.30 mm, lower 0.60 mm
   - density about 50–60%, angle threshold about 50°
   - supports only on sturdy areas, never on vines or lattice

   Remove supports **before** the final UV cure, warm the print first, and cut with flush cutters; never snap.
3. **Metal-ready versions.** Amy wants to try **electroplating or casting**. Every generator now has `--metal`:
   - thicker wires: openwork vines 1.0 mm, rails 0.85 mm; plumeria petal tips 0.85 mm
   - deeper engravings: heirloom lines 0.18 × 0.30 mm with a 0.55 mm background; openwork engraving 0.28 mm; ALOHA 0.40 mm deep
   - gyroid pendant: 0.8 mm walls and 1.4× cells, so pores stay ≥ 1 mm for plating

   Metal files get a `_metal` suffix. Still undecided: casting vs. plating for the rings, which sets the `--fit` value.
4. **CHITUBOX crashed** slicing a plate with every model on it (over 4 million triangles plus supports). Slice the sculpture alone and the jewelry separately; lower anti-aliasing while testing; consider simplified meshes (`decimate(mesh, 0.015)`) and re-verify islands after simplifying.

**How to work:**
- Generate with each script's CLI. For example:
  - `python3 openwork_band.py --size 7 --metal`
  - `python3 gyroid_pendant.py --metal`
- Check every part: watertight, single body, and no floating islands. The gyroid script re-slices and verifies itself.
- Show Amy preview renders and get approval before building final files (`plumeria-ring/render_preview.py` renders STLs without a GPU).
- `.ctb` files can be decoded (the scheme is in the lessons file) to check settings and layers for islands.
- Commit and push to the branch above. Keep files under 100 MB in git; zip larger ones.

**Possible next steps Amy has discussed:**
- metal-ready test prints and fit checks
- support-free earrings and a suncatcher using the gyroid approach
- polishing the lattice sculpture: lattice in the small flowers, a denser upper middle
