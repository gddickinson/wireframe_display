# Claude notes for wireframe_display

@INTERFACE.md

- Python files: keep under 500 lines (pygame_basicShapes.py is a pre-existing exception).
- Desktop modules import each other by name: add `pygame_wireframeDisplay/` to sys.path.
- Web engine changes: re-run `python scripts/make_engine_fixture.py --write` and `node web/static/wire_engine.test.mjs`.
