"""FastAPI web app for Wireframe Display: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8118, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.

Everything comes from the desktop code in pygame_wireframeDisplay/: the shape catalogue (shape_catalogue.py, the same
constructors and arguments as pygame_MatrixDisplay.py), the key table and scene operations (wireframe_scene.py), and the
shapes that need SciPy / scikit-image (convex hull, marching cubes), which are computed here and exported as JSON.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.engine_check import build_reference, run_check, shape_catalogue, wire_data  # also puts the package on sys.path

import pygame_parametricShapes  # noqa: E402
from wireframe_scene import KEY_BINDINGS  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
TITLE = "Wireframe Display"
CATALOGUE = shape_catalogue.entries()
SLUGS = [s["slug"] for s in CATALOGUE]
PYTHON_ONLY = [s["slug"] for s in CATALOGUE if s["python_only"]]
# Cross-check runs exported to the static site: all but marching cubes (subdivided it is ~650 kB; local app only).
CHECK_EXPORTED = [s for s in SLUGS if s != "marching_cubes"]

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary and counts for the home page."""
    groups = {}
    for s in CATALOGUE:
        groups[s["group"]] = groups.get(s["group"], 0) + 1
    return {"title": TITLE, "version": "1", "shapes": len(CATALOGUE), "groups": groups,
            "parametric_functions": len(shape_catalogue.PARAMETRIC), "keys": len(KEY_BINDINGS),
            "python_only": PYTHON_ONLY, "screen": [600, 600],
            "parametric_module": " ".join((pygame_parametricShapes.__doc__ or "").split())}


@app.get("/api/shapes")
def shapes():
    """Shape catalogue: slug, name, group, constructor and keyword arguments (as in pygame_MatrixDisplay.py)."""
    return {"shapes": CATALOGUE, "hull_seed": shape_catalogue.HULL_SEED, "check_exported": CHECK_EXPORTED,
            "not_included": ["teapot: loadOBJ('teapot.obj'), the file is not in the repository",
                             "MRI and super-resolution data: load local files from George's machine"]}


@app.get("/api/keys")
def keys():
    """Key bindings of the desktop viewer (wireframe_scene.KEY_BINDINGS)."""
    return {"keys": [{"key": k, "method": m, "args": a, "label": lbl} for k, m, a, lbl in KEY_BINDINGS]}


@app.get("/api/shape/{slug}")
def shape(slug: str):
    """Nodes, edges and faces of a catalogue shape built by the Python constructors (full precision)."""
    if slug not in SLUGS:
        raise HTTPException(404, f"unknown shape {slug!r}")
    return {"slug": slug, **wire_data(build_reference(slug))}


@app.get("/api/check/{slug}")
def check(slug: str):
    """Screen positions after a fixed sequence of viewer operations, computed by the Python scene (for Cross-check)."""
    if slug not in SLUGS:
        raise HTTPException(404, f"unknown shape {slug!r}")
    return run_check(slug)


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = (["/info", "/shapes", "/keys"] + [f"/shape/{s}" for s in PYTHON_ONLY]
                    + [f"/check/{s}" for s in CHECK_EXPORTED])
