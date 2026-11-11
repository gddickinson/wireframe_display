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
    pv = _run_script_until_event_loop(PKG / "pygame_MatrixDisplay.py", monkeypatch)["pv"]
    assert list(pv.wireframes) == ["octasphere"]
    assert np.allclose(pv.wireframes["octasphere"].findCentre()[:2], [300, 300])


def _centred_octasphere(cx=300.0, cy=300.0, cz=0.0):
    o = shapes.Octasphere()
    o.moveTo(cx, cy, cz)
    return o
