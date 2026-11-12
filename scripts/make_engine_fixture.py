"""Compute reference numbers with the repo's Python code for web/static/wire_engine.test.mjs.

Writes tests/fixtures/engine_fixture.json; tests/test_web.py recomputes it and checks the committed copy is current
(with a tolerance), so the JS engine is always compared against the current Python.

Run: python scripts/make_engine_fixture.py            (print a summary)
     python scripts/make_engine_fixture.py --write    (rewrite tests/fixtures/engine_fixture.json)
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.engine_check import CHECK_OPS, HEIGHT, WIDTH, build_reference, shape_catalogue  # noqa: E402

import pygame_basicShapes as basic  # noqa: E402
import wireframe_scene as ws  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_fixture.json"
SAMPLE_OBJ = ROOT / "tests" / "fixtures" / "sample.obj"
MAX_ROWS = 60
SCENE_SLUGS = ["cube", "tetra", "octa", "octasphere", "icosahedron", "ball", "letter", "grid", "fractal", "param_klein",
               "param_mobius_tube", "param_cornucopia"]


def sample(arr, max_rows=MAX_ROWS):
    """Every k-th row (at most max_rows) with its index, plus the column sums of all rows."""
    arr = np.asarray(arr, dtype=float)
    step = max(1, -(-len(arr) // max_rows))
    idx = list(range(0, len(arr), step))
    return {"n": len(arr), "idx": idx, "rows": arr[idx].tolist(), "sums": arr.sum(axis=0).tolist()}


def shape_cases():
    out = {}
    for e in shape_catalogue.entries():
        if e["python_only"]:
            continue
        wf = build_reference(e["slug"])
        out[e["slug"]] = {"spec": {"fn": e["fn"], "args": e["args"]}, "nodes": sample(wf.nodes[:, :3]),
                          "edges": len(wf.edges), "faces": len(wf.faces), "edges_head": [list(map(int, x)) for x in wf.edges[:12]]}
    return out


def scene_cases():
    out = {}
    for slug in SCENE_SLUGS:
        scene = ws.WireframeScene(WIDTH, HEIGHT)
        scene.addWireframe(slug, build_reference(slug))
        steps = []
        for method, args in CHECK_OPS:
            getattr(scene, method)(*args)
            wf = scene.wireframes[slug]
            last = (method, args) == tuple(CHECK_OPS[-1]) and len(steps) == len(CHECK_OPS) - 1
            steps.append({"op": method, "nodes": sample(wf.nodes, MAX_ROWS if last else 8), "history": len(wf.history)})
        out[slug] = {"steps": steps, "move_center": list(scene.moveCenter), "yoyo": scene.yoyoCounter,
                     "edges": len(wf.edges), "faces": len(wf.faces)}
    return out


def matrix_cases():
    out = []
    for th in (0.1, -0.7, 2.3):
        for name in ("rotateXMatrix", "rotateYMatrix", "rotateZMatrix"):
            out.append([name, [th], getattr(ws, name)(th).tolist()])
    out.append(["shearMatrix", [0.1, 0.2, -0.3, 0.1, 0.2, -0.3], ws.shearMatrix(0.1, 0.2, -0.3, 0.1, 0.2, -0.3).tolist()])
    out.append(["translationMatrix", [4, -3, 2], ws.translationMatrix(4, -3, 2).tolist()])
    out.append(["scaleMatrix", [2, 3, 4], ws.scaleMatrix(2, 3, 4).tolist()])
    return out


def subdivide_cases():
    octa = basic.Octahedron()
    octa.subdivide()
    osph = basic.Octasphere()
    osph.moveTo(300, 300, 0)
    osph.subdivide()
    return {"octa": {"nodes": sample(octa.nodes[:, :3]), "faces": len(octa.faces)},
            "octasphere_centred": {"nodes": sample(osph.nodes[:, :3]), "faces": len(osph.faces)}}


def fit_cases():
    out = {}
    for slug in ("cube", "octa", "param_torus"):
        scene = ws.WireframeScene(WIDTH, HEIGHT)
        scene.addWireframe(slug, build_reference(slug))
        scene.rotateAll("X", 0.4)
        scene.fitAll()
        out[slug] = sample(scene.wireframes[slug].nodes)
    return out


def obj_case():
    m = basic.loadOBJ(str(SAMPLE_OBJ))
    return {"nodes": m.nodes[:, :3].tolist(), "faces": [[int(i) for i in f[0]] for f in m.faces],
            "edges": [list(map(int, e)) for e in m.edges]}


def fixture():
    return {"shapes": shape_cases(), "scenes": scene_cases(), "ops": CHECK_OPS, "matrices": matrix_cases(),
            "subdivide": subdivide_cases(), "fit": fit_cases(), "obj": obj_case(), "keys": ws.KEY_BINDINGS,
            "rand": "frac(k * 0.6180339887498949), k = 1, 2, ... (stands in for random.random in FractalLandscape)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    fx = fixture()
    text = json.dumps(fx, separators=(",", ":"))
    if ap.parse_args().write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text + "\n")
        print(f"wrote {OUT.relative_to(ROOT)} ({len(text) / 1e3:.0f} kB)")
    else:
        print(f"{len(fx['shapes'])} shapes, {len(fx['scenes'])} scene runs, {len(text) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
