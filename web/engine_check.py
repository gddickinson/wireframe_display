"""Reference runs of the Python viewer maths, shared by the web API (/api/check) and scripts/make_engine_fixture.py.

A check builds a catalogue shape with the repo's own constructors, puts it in a WireframeScene (600 x 600, as the
desktop viewer), centres it and applies CHECK_OPS: the same scene operations the keys and the animation run. The JS
engine repeats the same steps and must land on the same screen positions.
"""
import random
import sys
from contextlib import contextmanager
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "pygame_wireframeDisplay"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import shape_catalogue  # noqa: E402
from wireframe_scene import WireframeScene  # noqa: E402

WIDTH = HEIGHT = 600
GOLDEN = 0.6180339887498949

# (method, args): keys q, a, z, =, t, h, RIGHT, then three animation ticks (spin + yoyo), m (subdivide), 0 (centre)
CHECK_OPS = [
    ["centerAll", []], ["rotateAll", ["X", 0.1]], ["rotateAll", ["Y", 0.1]], ["rotateAll", ["Z", 0.1]], ["scaleAll", [1.05]],
    ["shearAll", [[-0.1, 0, 0, 0, 0, 0]]], ["shearAll", [[0, 0.1, 0, 0, 0, 0]]], ["translateAll", [[10, 0, 0]]],
    ["animate_update", []], ["animate_update", []], ["animate_update", []], ["subdivideAll", []], ["centerAll", []],
]


def det_rand():
    """Deterministic stand-in for random.random() (the JS check uses the same sequence): frac(k * golden ratio)."""
    k = 0

    def r():
        nonlocal k
        k += 1
        return (k * GOLDEN) % 1.0
    return r


@contextmanager
def patched_random():
    original = random.random
    random.random = det_rand()
    try:
        yield
    finally:
        random.random = original


def build_reference(slug):
    """Catalogue shape built by Python; FractalLandscape draws its heights from det_rand()."""
    with patched_random():
        return shape_catalogue.build(slug)


def wire_data(wf, digits=None):
    rnd = (lambda v: round(float(v), digits)) if digits is not None else float
    return {"nodes": [[rnd(v) for v in row[:3]] for row in wf.nodes],
            "edges": [[int(a), int(b)] for a, b in wf.edges],
            "faces": [[int(i) for i in f[0]] for f in wf.faces]}


def run_check(slug, ops=CHECK_OPS):
    """Apply ops to the shape in a fresh scene; returns the projected (x, y) of every node after the last op."""
    scene = WireframeScene(WIDTH, HEIGHT)
    scene.addWireframe(slug, build_reference(slug))
    for method, args in ops:
        getattr(scene, method)(*args)
    wf = scene.wireframes[slug]
    return {"slug": slug, "ops": ops, "nodes": len(wf.nodes), "edges": len(wf.edges), "faces": len(wf.faces),
            "history": len(wf.history), "move_center": list(scene.moveCenter), "yoyo": scene.yoyoCounter,
            "points": [[round(float(x), 6), round(float(y), 6)] for x, y in wf.nodes[:, :2]],
            "rand": "frac(k * 0.6180339887498949), k = 1, 2, ...", "source": "wireframe_scene.WireframeScene (NumPy)"}
