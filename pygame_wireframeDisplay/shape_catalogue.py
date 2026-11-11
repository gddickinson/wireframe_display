"""The demo shapes of the matrix wireframe viewer as data: which constructor, with which arguments.

The entries mirror the model table in pygame_MatrixDisplay.py's __main__ (same constructors and arguments) plus every
surface function in pygame_parametricShapes.py. The web viewer builds each shape in the browser from the same
(fn, args) spec; shapes marked python_only need SciPy / scikit-image and are served as precomputed nodes and edges.
Not included: teapot (teapot.obj is not in the repository), MRI and super-resolution data (local files).
"""
import inspect

import numpy as np

import pygame_basicShapes as basic
import pygame_parametricShapes as param

X, Y, Z = (300, 300, 250)  # as in pygame_MatrixDisplay.py
W, H, D = (50, 50, 50)
U, V = (30, 30)

PARAMETRIC = ["sphere", "torus", "funnel", "noseCone", "helix_3D", "klein", "mobius_tube", "helix", "sineSurface",
              "eightSurface", "ellipsoid", "hyperOctahedron", "geodesic", "cornucopia"]

# slug, name, group, constructor in pygame_basicShapes, keyword arguments, python_only
SHAPES = [
    ("cube", "Cube", "solid", "Cuboid", {"args1": [X, Y, Z], "args2": [W, H, D]}, False),
    ("tetra", "Tetrahedron", "solid", "Tetrahedron", {"args1": [0, 0, 0], "edgeLength": 100}, False),
    ("octa", "Octahedron", "solid", "Octahedron", {}, False),
    ("octasphere", "Octasphere", "solid", "Octasphere", {}, False),
    ("icosahedron", "Icosahedron", "solid", "Icosahedron", {}, False),
    ("ball", "Spheroid", "solid", "Spheroid", {"args1": [X, Y, Z], "args2": [H, W, D], "resolution": 10}, False),
    ("letter", "Letter A", "solid", "letter", {}, False),
    ("grid", "Horizontal grid", "terrain", "HorizontalGrid", {"args1": [X, Y, Z], "args2": [10, 10], "args3": [10, 20]}, False),
    ("fractal", "Fractal landscape", "terrain", "FractalLandscape",
     {"origin": [X, Y, Z], "dimensions": [400, 400], "iterations": 5, "height": 40}, False),
    ("marching_cubes", "Marching cubes", "computed", "marchingCubes", {}, True),
    ("hull", "Convex hull of torus points", "computed", "convexHull", {"torusPts": True}, True),
] + [(f"param_{f.lower()}", f"Parametric {f}", "parametric", "parametric_surface",
      {"slices": U, "stacks": V, "func": f}, False) for f in PARAMETRIC]

HULL_SEED = 0  # convexHull samples random torus points; the web shows the hull for this NumPy seed


def entries():
    """Catalogue as dicts (slug, name, group, fn, args, python_only, doc)."""
    out = []
    for slug, name, group, fn, args, py in SHAPES:
        target = getattr(param, args["func"]) if fn == "parametric_surface" else getattr(basic, fn)
        doc = inspect.getdoc(target) or ""
        if fn == "parametric_surface":
            doc = "x, y, z = " + args["func"] + "(u, v): " + _source_body(target)
        out.append({"slug": slug, "name": name, "group": group, "fn": fn, "args": args, "python_only": py,
                    "doc": " ".join(doc.replace('"', "").split())})
    return out


def _source_body(func):
    """The formula lines of a parametric surface function (its source without def/return)."""
    lines = inspect.getsource(func).splitlines()[1:]
    return "; ".join(ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("return"))


def build(slug):
    """Build the wireframe for a catalogue slug with the repo's own constructors."""
    spec = next(s for s in SHAPES if s[0] == slug)
    _, _, _, fn, args, _ = spec
    if fn == "parametric_surface":
        return basic.parametric_surface(args["slices"], args["stacks"], getattr(param, args["func"]))
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in args.items()}
    if fn == "convexHull":
        state = np.random.get_state()
        np.random.seed(HULL_SEED)
        try:
            return basic.convexHull(**kwargs)
        finally:
            np.random.set_state(state)
    return getattr(basic, fn)(**kwargs)
