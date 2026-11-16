# Shared tools

Tools that work on any design. Each one prints its own `--help`.

| Tool | What it does |
|---|---|
| `render_views.py` | Shaded multi-view preview renders of one or more STLs, used for approval previews. It needs no GPU. Use `--up z` for sculptures standing on the plate and `--up y` for flat pieces such as suncatchers. `--close=X,Y,Z` adds a close-up. |
| `slice_check.py` | A printer-style slice check of an STL. It lists islands (layer pieces with nothing below them), the largest overhang ledge, and the flat downward area. `--stand --lean 0 30 45` tries a flat piece stood up and leaned back. Use `--layer 0.1 --xy 0.1` for big flat pieces; the defaults (0.05) suit jewelry and sculptures. |
| `ctb_check.py` | Reads a CHITUBOX `.ctb` file. It prints the settings (printer, resolution, layer height, exposures, anti-aliasing, resin, time) and checks every layer for islands, counted the way the slicer does, with how many layers each one floats. |

Design-specific checks live with their designs:
- `gyroid_pendant.verify`, which `slice_check.py` follows.
- `lattice_plumeria.find_islands`, for SDF volumes.
- Each sculpture's `final_build.py`, which runs its own island loop and patches.

Typical flow for a new piece:
1. Use the generator's `--preview` mode, then `render_views.py` for the approval panel.
2. Once Amy approves, run the final build, then `slice_check.py` on the final STL.
3. After Amy slices it in CHITUBOX, run `ctb_check.py` on her `.ctb`.
