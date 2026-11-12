# SESSION_LOG.md

## 2026-11-11: web interface, static GitHub Pages site, bug fixes (branch `web-interface`)
Done (Claude, for George; not pushed):
- Fixed bugs in the desktop code, each in its own commit with a regression test in `tests/test_desktop.py`
  (headless, SDL dummy driver):
  1. `marchingCubes()` / `MRI()` used `measure.marching_cubes_lewiner`, removed in scikit-image 0.19 → `measure.marching_cubes`.
  2. `pygame_MatrixDisplay.py` crashed at start-up: its model table built every model, including
     `loadOBJ('teapot.obj')` (file not in the repo). Models are now built on demand.
  3. `Wireframe.subdivide()` (key M) put new vertices on the unit sphere at the origin (garbage for the default
     octasphere, radius 20 at the screen centre) and crashed on quad meshes. Now uses the shape's own centre and mean
     radius; non-triangle meshes are left unchanged.
  4. `Wireframe.update()` saved history by reference; rotate/scale/shear modify nodes in place, so the trail only
     showed translations. Now stores a copy.
  5. `loadOBJ` split lines on single spaces, so OBJ lines padded with several spaces (the code comment's own example)
     raised ValueError. Now splits on any whitespace and skips blank lines (also in `pygame_loadOBJ_test.py`).
  6. `pygame_basicShapes.py` run as a script called undefined `Hexahedron()`; `pygame_Matrix-display.py` had a
     SyntaxError (unclosed dict), undefined `cube_nodes`, Python 2 `itervalues()`.
- Refactor (behaviour unchanged, verified node-for-node against the old `ProjectionViewer` for a key sequence plus 45
  animation ticks): scene operations, matrices and key table moved to Pygame-free `wireframe_scene.py`;
  `ProjectionViewer` subclasses `WireframeScene`. Demo shapes catalogued in `shape_catalogue.py`.
- Web: FastAPI app (`web/`, `run_web.py`, port 8118), JS engine (`wire_engine.js`, `shapes_engine.js`) node-tested
  against `tests/fixtures/engine_fixture.json` (from `scripts/make_engine_fixture.py`), pages home / viewer / shapes /
  cross-check / about, static build + Pages workflow, README section with screenshots, INTERFACE.md, CLAUDE.md.

Needs George's decision (not changed):
- `ProjectionViewer.timedAction` removes wireframes 10 s after creation by `wireframe.name`; the name matches the scene
  key only for cube / tetra / grid (they vanish), otherwise "not found" is printed every frame. Intended? If not,
  remove the call in `run()`; if yes, remove by dict key.
- `pygame_wireframe_test.py` / `pygame_wireframe_examples.py` import `wireframe`, `wireframeDisplay`, `basicShapes`
  (not in this repo; the names match Peter Collingridge's Pygame 3D tutorial). Delete, or port to the matrix modules?
- `pygame_Matrix-wireframe.py`: translate/scale/findCentre/rotateX/Y/Z still use `node.x` on numpy rows (would raise
  if called; its demo does not call them).
- `loadOBJ` splits polygons into consecutive triangles (i, i+1, i+2), not a fan (0, i, i+1): fine for quads' outlines,
  overlapping for pentagons and larger. Left as is (the web port does the same).
- Credit: much of the structure (Wireframe / ProjectionViewer / key_to_function / Spheroid / FractalLandscape) appears
  to follow Peter Collingridge's "3D graphics with Pygame" tutorial; the site does not claim this. Add a credit?

## 2026-11-11 (later): follow-up fixes after merge (branch `web-interface`, fast-forwarded to main)
George: "fix any other problems you come across". Each fix has its own commit and a failing-first test in tests/test_desktop.py:
- timedAction: now `WireframeScene.timedAction(now)`, off by default; `python pygame_MatrixDisplay.py --demo [SECONDS]`
  starts a timed demo (each model fitted with the new `fitAll`, shown for SECONDS, then the next; removal by scene
  key, nothing printed). Ported to the web engine (node-tested) with a "Timed demo" check box; Fit now uses `fitAll`.
- `pygame_loadOBJ_test.py` did not compile (empty `__main__` block) -> `pass`; test that every script compiles.
- `pygame_wireframe_examples.py` / `pygame_wireframe_test.py` are copies of examples.py / testAnimation.py from
  petercollingridge/Pygame3D and need its modules (no licence file there, so not copied) -> moved to `_archive/` with
  a README; test that no desktop script imports a missing module.
- Credit to Peter Collingridge's "3D Graphics with Pygame" tutorial and Pygame3D added to README and About
  (derivation checked against https://github.com/petercollingridge/Pygame3D: same shapes and matrix code).
- `loadOBJ`: polygons now split into a fan (0, i, i+1); consecutive triples overlapped for 5+ corners (Python + JS).
- `pygame_Matrix-wireframe.py`: translate / scale / findCentre / rotateX/Y/Z fixed for numpy rows.
Checks: pytest tests/ 54 passed (test_web 37); node 56 checks; screenshots static (8 routes) and against
run_web.py on 8118 (6 routes) all ok, phone overflow 0px.
Still for George: Pygame3D has no licence file, yet several shapes and the matrix code derive from it (now credited).
