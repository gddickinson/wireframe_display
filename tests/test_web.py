"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- Wireframe Display API tests --------------------------------------------------------------------------------------
import json  # noqa: E402

import numpy as np  # noqa: E402

from web.app import CATALOGUE  # noqa: E402
from web.engine_check import run_check  # noqa: E402
from wireframe_scene import KEY_BINDINGS  # noqa: E402


def test_catalogue_and_keys_come_from_python():
    d = client.get("/api/shapes").json()
    assert [s["slug"] for s in d["shapes"]] == [s["slug"] for s in CATALOGUE]
    assert {s["slug"] for s in d["shapes"] if s["python_only"]} == {"hull", "marching_cubes"}
    k = client.get("/api/keys").json()["keys"]
    assert [(x["key"], x["method"], x["args"]) for x in k] == [(a, b, c) for a, b, c, _ in KEY_BINDINGS]


def test_python_only_shapes_served():
    h = client.get("/api/shape/hull").json()
    assert len(h["nodes"]) == 500 and len(h["edges"]) > 500
    assert client.get("/api/shape/hull").json() == h  # fixed seed: same hull every time
    assert client.get("/api/shape/nope").status_code == 404


def test_check_matches_scene():
    r = client.get("/api/check/cube").json()
    ref = run_check("cube")
    assert r["points"] == ref["points"] and r["nodes"] == 8 and r["edges"] == 24
    assert np.allclose(np.mean(r["points"], axis=0), [300, 300])  # last op is centerAll


def test_engine_fixture_is_current():
    """tests/fixtures/engine_fixture.json (read by wire_engine.test.mjs) still matches the Python code."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_engine_fixture
    saved = json.loads(make_engine_fixture.OUT.read_text())
    now = json.loads(json.dumps(make_engine_fixture.fixture()))

    def walk(a, b, path="fixture"):
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a:
                walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, float) or isinstance(b, float):
            assert np.isclose(a, b, rtol=1e-9, atol=1e-9), (path, a, b)
        else:
            assert a == b, (path, a, b)
    walk(saved, now)
