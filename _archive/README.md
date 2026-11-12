# _archive

Scripts kept for reference that do not run in this repository.

| File | What it is |
|---|---|
| `pygame_wireframe_examples.py` | A Python 3 copy of `examples.py` from Peter Collingridge's [Pygame3D](https://github.com/petercollingridge/Pygame3D) (the code of his [3D Graphics with Pygame](https://www.petercollingridge.co.uk/tutorials/3d/pygame/) tutorial). |
| `pygame_wireframe_test.py` | A copy of `testAnimation.py` from the same repository (an animated ball over a grid). |

Both import that repository's own modules (`wireframe`, `wireframeDisplay`, `basicShapes`), which are not part of
this project; the matrix code here (`pygame_wireframeDisplay/`) is a separate, reworked version with a different API
(e.g. `scaleMatrix(sx, sy, sz)`, no `WireframeGroup` / `WireframeViewer`). To run them, put them next to a checkout of
Pygame3D (its modules are written for Python 2). Pygame3D has no licence file, so its modules are not copied here.
