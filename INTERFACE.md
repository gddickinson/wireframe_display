# INTERFACE.md: navigation map for wireframe_display

## Desktop code: `pygame_wireframeDisplay/` (flat scripts, imported by module name; run from that folder)
| File | Contents |
|---|---|
| `pygame_MatrixWireframe.py` | `Wireframe`: nodes as (x, y, z, 1) rows, edges, faces, history; `transform` (row vectors), `scale`/`shear`/`rotate` about a centre, `moveTo`, `changeShape`, `update` (records history), `subdivide` (onto the shape's own sphere) |
| `wireframe_scene.py` | **Pygame-free** scene logic: matrix builders (`translationMatrix`, `scaleMatrix`, `rotateX/Y/ZMatrix`, `shearMatrix`, `perspectiveMatrix`), `KEY_BINDINGS`, `WireframeScene` (translateAll, scaleAll, rotateAll, shearAll, centerAll, moveAll, subdivideAll, spin, yoyo, animate_update, apply_key), `project` |
| `pygame_MatrixDisplay.py` | `ProjectionViewer(WireframeScene)`: Pygame window, event loop (keys, 50 ms / 500 ms timers), `display()`, `timedAction`; `__main__` demo model table |
| `pygame_basicShapes.py` | shape constructors: `Cuboid`, `Tetrahedron`, `Octahedron`, `Spheroid`, `HorizontalGrid`, `FractalLandscape`, `letter`, `Icosahedron`, `Octasphere`/`get_octasphere` (pyrr), `parametric_surface`, `loadOBJ`, `marchingCubes` (skimage), `convexHull` (scipy), `MRI`, `loadNIKONFile`, `loadImage*` (local files) |
| `pygame_parametricShapes.py` | 14 surface functions f(u, v) → (x, y, z) |
| `shape_catalogue.py` | **Pygame-free** catalogue of the demo shapes as (slug, name, group, constructor, kwargs); `entries()`, `build(slug)` |
| `pygame_display.py`, `pygame_wireframe.py` | earliest step: `Node`/`Edge` objects, trigonometric rotation |
| `pygame_Matrix-display.py`, `pygame_Matrix-wireframe.py` | first matrix step (hyphenated: run as scripts, not importable) |
| `pygame_loadOBJ_test.py` | matplotlib OBJ plotting experiment |

## `_archive/`
Tutorial scripts copied from petercollingridge/Pygame3D that need its modules (see `_archive/README.md`).

## Web: `web/`
| File | Contents |
|---|---|
| `web/app.py` | FastAPI: `/api/info`, `/api/shapes`, `/api/keys`, `/api/shape/{slug}`, `/api/check/{slug}`; `STATIC_ENDPOINTS`, `CHECK_EXPORTED` |
| `web/engine_check.py` | `CHECK_OPS`, `det_rand`, `build_reference(slug)`, `run_check(slug)`, `wire_data` (shared by API and fixture script) |
| `web/static/wire_engine.js` | JS port of `Wireframe`, matrices, `KEY_BINDINGS`, `Scene` (= WireframeScene), `project`, `keyName` |
| `web/static/shapes_engine.js` | JS port of the shape constructors, surfaces, octasphere quaternions, `loadOBJ`; `build(spec)`, `fromData` |
| `web/static/render.js` | canvas `draw` (port of `display()`), `fit`, `makeShape`, `newScene`, `catalogue` |
| `web/static/{home,viewer,gallery,check,about}.js` | pages (routes `''`, `viewer`, `shapes`, `check`, `about`) |
| `web/static/core.js`, `style.css` | shared web kit (router, api(), helpers, theme), unchanged |
| `web/static/main.js`, `static_api.js`, `index.html`, `style_app.css`, `icon.svg` | entry point, static-site data layer, shell, accent/page styles |
| `web/static/wire_engine.test.mjs` | node test: JS engine vs `tests/fixtures/engine_fixture.json` |

## Scripts and tests
| File | Contents |
|---|---|
| `run_web.py` | starts the web app on 127.0.0.1:8118 |
| `scripts/build_static.py` | exports the static site to `site/` |
| `scripts/screenshots.py` | Playwright smoke test and README images (`docs/img/`) |
| `scripts/make_engine_fixture.py` | writes `tests/fixtures/engine_fixture.json` from the Python code |
| `tests/test_web.py` | API, static build, node engine test, fixture-is-current |
| `tests/test_desktop.py` | headless regression tests for the Pygame code |
| `tests/fixtures/` | `engine_fixture.json`, `sample.obj` (OBJ parser test input) |
| `.github/workflows/pages.yml` | tests + build + GitHub Pages deploy on push to main |

## How it connects
`ProjectionViewer` → inherits `WireframeScene` → drives `Wireframe` objects built by `pygame_basicShapes`.
`web/app.py` → `engine_check` → `shape_catalogue` + `wireframe_scene` (no Pygame). The browser builds shapes with
`shapes_engine.js` from the same catalogue specs and moves them with `wire_engine.js`.
