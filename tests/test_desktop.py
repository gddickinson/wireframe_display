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


def _centred_octasphere(cx=300.0, cy=300.0, cz=0.0):
    o = shapes.Octasphere()
    o.moveTo(cx, cy, cz)
    return o
