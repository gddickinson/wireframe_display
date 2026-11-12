"""Regression tests for the desktop (Pygame) code in pygame_wireframeDisplay/.

Runs headless (SDL dummy video driver). Needs numpy, scipy, pyrr, scikit-image and pygame:
    pip install -r requirements.txt pytest
Run: python -m pytest -q tests/test_desktop.py
"""
import os
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "pygame_wireframeDisplay"
sys.path.insert(0, str(PKG))

import pygame_basicShapes as shapes  # noqa: E402
import pygame_MatrixWireframe as wf  # noqa: E402


def test_marching_cubes_runs_with_current_scikit_image():
    """marching_cubes_lewiner was removed in scikit-image 0.19; the demo shape must still build."""
    model = shapes.marchingCubes()
    assert len(model.nodes) > 100 and len(model.faces) > 100


class _StopLoop(Exception):
    pass


def _run_script_until_event_loop(path, monkeypatch):
    """Execute a viewer script as __main__ and stop it when it first polls for events; returns its globals."""
    import pygame
    def stop():
        raise _StopLoop
    monkeypatch.setattr(pygame.event, "get", stop)
    g = {"__name__": "__main__", "__file__": str(path)}
    with pytest.raises(_StopLoop):
        exec(compile(Path(path).read_text(), str(path), "exec"), g)
    return g


def test_matrix_display_starts_without_optional_files(tmp_path, monkeypatch):
    """Running pygame_MatrixDisplay.py must not need teapot.obj (not in the repo) or build every model up front."""
    monkeypatch.chdir(tmp_path)  # no teapot.obj here
    monkeypatch.setattr(sys, "argv", ["pygame_MatrixDisplay.py"])
    pv = _run_script_until_event_loop(PKG / "pygame_MatrixDisplay.py", monkeypatch)["pv"]
    assert list(pv.wireframes) == ["octasphere"]
    assert np.allclose(pv.wireframes["octasphere"].findCentre()[:2], [300, 300])


def _centred_octasphere(cx=300.0, cy=300.0, cz=0.0):
    o = shapes.Octasphere()
    o.moveTo(cx, cy, cz)
    return o


def test_subdivide_keeps_shape_where_it_is():
    """New vertices go onto the shape's own sphere, not the unit sphere at the origin."""
    o = _centred_octasphere()
    centre = o.findCentre()[:3]
    r0 = np.linalg.norm(o.nodes[:, :3] - centre, axis=1).mean()
    o.subdivide()
    d = np.linalg.norm(o.nodes[:, :3] - centre, axis=1)
    assert len(o.nodes) > 1000
    assert np.allclose(d, r0, rtol=0.05), (d.min(), d.max(), r0)


def test_subdivide_unit_octahedron_unchanged_behaviour():
    """For a unit shape at the origin the result is the classic subdivision (all vertices on the unit sphere)."""
    o = shapes.Octahedron()
    o.subdivide()
    assert len(o.faces) == 32
    assert np.allclose(np.linalg.norm(o.nodes[:, :3], axis=1), 1.0, atol=1e-6)


def test_subdivide_skips_non_triangle_meshes():
    """Pressing M with a cube (quad faces) used to crash the viewer with a ValueError."""
    c = shapes.Cuboid(args1=(300, 300, 250), args2=(50, 50, 50))
    before = c.nodes.copy()
    c.subdivide()
    assert np.array_equal(c.nodes, before) and len(c.faces) == 6


def test_history_keeps_previous_positions():
    """update() stores the old nodes; rotate/scale/shear change nodes in place, so it must store a copy."""
    t = shapes.Tetrahedron(args1=(300, 300, 0), edgeLength=100)
    start = t.nodes.copy()
    c = t.findCentre()
    t.update("rotate", c, np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]]))
    assert np.allclose(t.history[-1], start)
    assert not np.allclose(t.nodes, start)


def test_load_obj_accepts_repeated_spaces(tmp_path):
    """OBJ files often pad with several spaces ("v  30.2180 89.5757 -76.8089", the example in the code's own comment)."""
    p = tmp_path / "quad.obj"
    p.write_text("# test\n\nv  0.0 0.0  0.0\nv 1.0  0.0 0.0\nv 1.0 1.0 0.0\nv  0.0 1.0 0.0\nvn 0 0 1\nf 1/1/1  2/2/1 3/3/1 4/4/1\n")
    m = shapes.loadOBJ(str(p))
    assert m.nodes.shape == (4, 4) and len(m.faces) == 2


def test_basic_shapes_module_runs_as_script():
    """`python pygame_basicShapes.py` referenced an undefined Hexahedron()."""
    runpy.run_path(str(PKG / "pygame_basicShapes.py"), run_name="__main__")


def test_early_matrix_display_script_is_valid_python():
    """pygame_Matrix-display.py (an earlier tutorial step) had an unclosed dict and Python 2 itervalues()."""
    import pygame
    ns = runpy.run_path(str(PKG / "pygame_Matrix-display.py"), run_name="not_main")
    pv = ns["ProjectionViewer"](400, 300)
    cube = wf.Wireframe()
    cube.addNodes(np.array(ns["cube_nodes"]))
    pv.addWireframe("cube", cube)
    pv.translateAll([10, 0, 0])
    assert np.allclose(pv.wireframes["cube"].nodes[0, :3], [60, 50, 50])
    pygame.quit()


def test_viewer_keys_drive_the_shared_scene_operations():
    """ProjectionViewer's keys come from wireframe_scene.KEY_BINDINGS and give the same nodes as WireframeScene."""
    import pygame
    import pygame_MatrixDisplay as md
    from wireframe_scene import KEY_BINDINGS, WireframeScene
    pv, ref = md.ProjectionViewer(600, 600), WireframeScene(600, 600)
    for s in (pv, ref):
        s.addWireframe("o", shapes.Octasphere())
        s.centerAll()
    assert len(md.key_to_function) == len(KEY_BINDINGS)
    for key, *_ in KEY_BINDINGS:
        if key != "9":
            md.key_to_function[getattr(pygame, "K_" + key)](pv)
            ref.apply_key(key)
    for _ in range(5):
        pv.animate_update()
        ref.animate_update()
    pv.display()
    assert np.array_equal(pv.wireframes["o"].nodes, ref.wireframes["o"].nodes)
    pygame.quit()


# --- timed demo (ProjectionViewer.timedAction) ----------------------------------------------------------------------
from datetime import datetime, timedelta  # noqa: E402


def _aged(wf, seconds):
    wf.initTime_raw = datetime.now() - timedelta(seconds=seconds)
    return wf


def test_timed_action_is_off_by_default(capsys):
    """Shapes stay on screen however old they are, and nothing is printed every frame."""
    from wireframe_scene import WireframeScene
    scene = WireframeScene(600, 600)
    scene.addWireframe("cube", _aged(shapes.Cuboid((300, 300, 0), (50, 50, 50)), 60))
    scene.addWireframe("octasphere", _aged(shapes.Octasphere(), 60))  # name '' did not match its key
    for _ in range(3):
        assert scene.timedAction(datetime.now()) == []
    assert list(scene.wireframes) == ["cube", "octasphere"]
    assert capsys.readouterr().out == ""


def test_timed_demo_cycles_through_models(capsys):
    from wireframe_scene import WireframeScene
    scene = WireframeScene(600, 600)
    models = [("tetra", lambda: shapes.Tetrahedron((0, 0, 0), 100)), ("octa", shapes.Octahedron)]
    scene.startTimedDemo(models, lifetime=5)
    assert list(scene.wireframes) == ["tetra"]
    assert scene.timedAction(datetime.now()) == []  # younger than 5 s
    later = datetime.now() + timedelta(seconds=6)
    assert scene.timedAction(later) == ["tetra"] and list(scene.wireframes) == ["octa"]
    octa = scene.wireframes["octa"]
    assert np.allclose(octa.findCentre()[:2], [300, 300])
    assert np.isclose(np.linalg.norm(octa.nodes[:, :3] - octa.findCentre()[:3], axis=1).max(), 230)
    _aged(octa, 6)
    assert scene.timedAction(datetime.now()) == ["octa"] and list(scene.wireframes) == ["tetra"]
    assert capsys.readouterr().out == ""


def test_matrix_display_demo_option(monkeypatch):
    """python pygame_MatrixDisplay.py --demo 4 starts the timed demo with the first model."""
    monkeypatch.setattr(sys, "argv", ["pygame_MatrixDisplay.py", "--demo", "4"])
    pv = _run_script_until_event_loop(PKG / "pygame_MatrixDisplay.py", monkeypatch)["pv"]
    assert pv.lifetime == 4 and len(pv.demo) >= 10 and list(pv.wireframes) == [pv.demo[0][0]]
    assert "teapot" not in [n for n, _ in pv.demo]



def test_every_desktop_script_compiles():
    """pygame_loadOBJ_test.py ended in an `if __name__ == "__main__":` whose body was all comments (IndentationError)."""
    for path in sorted(PKG.glob("*.py")):
        compile(path.read_text(), str(path), "exec")
