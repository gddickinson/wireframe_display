// About: what the viewer does (from the code), the matrices, the key table, how to run the desktop and web apps, credits.
import { route, app, api, esc, list } from './core.js';

route('about', async () => {
  const [{ keys }, info] = await Promise.all([api('/keys'), api('/info')]);
  const kname = k => ({ LEFT: '←', RIGHT: '→', UP: '↑', DOWN: '↓', EQUALS: '=', MINUS: '−' }[k] || k.toUpperCase());
  app().innerHTML = `<p class="kicker">About</p><h1 class="glow">How the viewer works</h1>
  <p class="lead"><a href="https://github.com/gddickinson/wireframe_display">wireframe_display</a> "displays 3D projections": a set of
  Python/Pygame scripts that build 3D wireframes (nodes and edges), move them with 4 × 4 matrices and draw them on a 600 × 600 window.
  This site runs the same maths in your browser.</p>

  <h2>Nodes, edges, faces</h2>
  <p>A wireframe (<code>pygame_MatrixWireframe.Wireframe</code>) keeps its nodes as rows <span class="mono">(x, y, z, 1)</span> in
  screen pixels, with y pointing down. Faces are lists of node indices; adding a face adds its outline to the edges, and only edges and
  nodes are drawn. Every operation first saves the old nodes to a history of up to 19 positions, which the optional trail draws.</p>

  <h2>The matrices</h2>
  <p>Translation is applied to the node rows, <span class="eq">v′ = v · T</span>, with T the identity plus
  <span class="mono">(dx, dy, dz)</span> in its bottom row. Rotation, scaling and shear act about a centre c:
  <span class="eq">v′ = c + M · (v − c)</span>. Rotations turn each shape about its own centre; zoom scales about the centre of the
  screen. For example the rotation about the x axis is</p>
  <pre>        | 1    0      0     0 |
Rx(θ) = | 0  cos θ −sin θ   0 |
        | 0  sin θ  cos θ   0 |
        | 0    0      0     1 |</pre>
  <p>The shear keys use a symmetric shear: one value h fills both mirrored off-diagonal places (h<sub>xy</sub> = h<sub>yx</sub>, and so on).</p>

  <h2>The projection</h2>
  <p>The viewer draws each node at its (x, y): an orthographic projection straight down the z axis, so depth changes nothing on screen
  except through the rotations. The code also defines a perspective matrix (<code>perspectiveMatrix(c)</code>), but the viewer never
  uses it for drawing. The animation's "yo-yo" fakes depth by scaling the scene by 0.85 for 20 ticks and by 1.1762 for the next 20.</p>

  <h2>Keys</h2>
  <div class="scroll"><table><tr><th>Key</th><th>Operation</th><th>What it does</th></tr>
  ${list(keys, k => `<tr><td><kbd>${esc(kname(k.key))}</kbd></td><td class="mono">${esc(k.method)}(${esc(k.args.map(a => JSON.stringify(a)).join(', '))})</td><td>${esc(k.label)}</td></tr>`)}</table></div>
  <p>The desktop viewer also has switches with no key (set in the code): animation (every 50 ms: <code>spin()</code>, three small
  rotations about shifted centres, then the yo-yo), the history trail with optional colour shift, shrinking the trail every 500 ms,
  a random node colour every 50 ms and random colours for every edge and node. The web viewer shows them as check boxes.</p>

  <h2>What the web version covers</h2>
  <p><code>wire_engine.js</code> is a port of <code>Wireframe</code> and of the scene operations, which now live in the Pygame-free
  <code>wireframe_scene.py</code> (the desktop <code>ProjectionViewer</code> inherits them). <code>shapes_engine.js</code> ports the
  ${info.shapes - info.python_only.length} shapes the browser can build, including the octasphere's quaternion maths from pyrr.
  The convex hull (SciPy) and marching cubes (scikit-image) are computed by the Python code and loaded as data. Everything is tested
  against numbers from the Python (see <a href="#/check">Cross-check</a>).</p>
  <p>Web additions: drag to rotate (rotateAll about Y and X), the mouse wheel (the = and − zoom), Fit (centre, then scale so the shape
  fills the view; the desktop app only centres, so unit-sized shapes start one pixel wide), Add (several shapes at once), a node-size
  slider, and opening your own OBJ file (parsed in the browser by a port of <code>loadOBJ</code>).
  The timed demo (<code>python pygame_MatrixDisplay.py --demo</code>: a new shape every 10 s) is a check box; the web demo
  cycles through the shapes the browser builds. Not ported: the teapot (teapot.obj is not in the repository), and the MRI, super-resolution and image loaders, which read files from the author's machine.</p>

  <h2>Run it yourself</h2>
  <h3>Desktop app (Pygame)</h3>
  <pre>pip install numpy scipy pyrr scikit-image pygame
cd pygame_wireframeDisplay
python pygame_MatrixDisplay.py      # the matrix viewer, starts with the octasphere</pre>
  <p>Change <code>wf_name</code> at the bottom of <code>pygame_MatrixDisplay.py</code> to show another model. The folder also holds
  earlier steps of the same viewer: <code>pygame_display.py</code> (node objects, trigonometric rotation) and
  <code>pygame_Matrix-display.py</code> (first matrix version).</p>
  <h3>Web app (this site, with the Python API)</h3>
  <pre>pip install -r requirements-web.txt
python run_web.py          # http://127.0.0.1:8118</pre>
  <p>The local app serves the shape catalogue and key table from the Python modules, computes the hull and marching-cubes shapes and
  runs the Python side of the cross-check live. The static GitHub Pages site is built by <code>scripts/build_static.py</code>.</p>

  <h2>Credits</h2>
  <p>wireframe_display by George Dickinson, MIT licence, based on Peter Collingridge's
  <a href="https://www.petercollingridge.co.uk/tutorials/3d/pygame/">3D Graphics with Pygame</a> tutorial and its example code
  (<a href="https://github.com/petercollingridge/Pygame3D">Pygame3D</a>): the node/edge wireframe, the matrix transformations, the
  projection viewer and key table, and the cuboid, spheroid, grid and fractal-landscape shapes follow it. The octasphere code credits
  <a href="https://github.com/prideout/svg3d">prideout/svg3d</a>. A related project by the same author is
  <a href="https://github.com/gddickinson/wireframeDisplay">wireframeDisplay</a> (a PyQt/OpenGL scene viewer).
  Web version built with AI assistance from <a href="https://claude.com/">Claude (Anthropic)</a>.</p>`;
});
