// Shapes: every catalogue entry drawn once (fitted and tilted), with its constructor call; click to open in the viewer.
import { route, app, esc, list } from './core.js';
import { catalogue, newScene, makeShape, fit, setupCanvas, draw } from './render.js';

const GROUP_TEXT = {
  solid: 'Polyhedra and closed solids from pygame_basicShapes.py.',
  terrain: 'Grids: a flat horizontal grid and a midpoint-displacement fractal landscape (random heights; a new one each time).',
  computed: 'Built with SciPy (ConvexHull) and scikit-image (marching cubes): computed by the Python code and loaded from the API.',
  parametric: 'parametric_surface(30, 30, f): f(u, v) sampled at u = iπ/30, v = 2jπ/30; every function in pygame_parametricShapes.py.',
};

route('shapes', async () => {
  const { shapes, not_included } = await catalogue();
  const groups = [...new Set(shapes.map(s => s.group))];
  app().innerHTML = `<p class="kicker">Shapes</p><h1 class="glow">The demo shapes</h1>
  <p class="lead">${shapes.length} shapes from the desktop code, with the constructor call that makes them. Each is centred and scaled to
  fit, then tilted a little (rotateAll X 0.5, Y 0.6) so it reads as 3D.</p>
  ${list(groups, g => `<h2>${esc(g[0].toUpperCase() + g.slice(1))}</h2><p class="mute">${esc(GROUP_TEXT[g] || '')}</p>
    <div class="grid gal">${list(shapes.filter(s => s.group === g), s => `<a class="card" href="#/viewer?shape=${s.slug}">
      <canvas data-slug="${s.slug}" role="img" aria-label="${esc(s.name)} wireframe"></canvas>
      <b>${esc(s.name)}</b><span class="mono">${esc(s.fn)}${s.fn === 'parametric_surface' ? `(30, 30, ${esc(s.args.func)})` : '()'}</span>
      ${s.doc ? `<span>${esc(s.doc)}</span>` : ''}</a>`)}</div>`)}
  <h2>Not here</h2><ul>${list(not_included, t => `<li>${esc(t)}</li>`)}</ul>`;
  for (const c of document.querySelectorAll('canvas[data-slug]')) {
    const entry = shapes.find(s => s.slug === c.dataset.slug);
    try {
      const scene = newScene(); scene.addWireframe(entry.slug, await makeShape(entry));
      fit(scene, 250); scene.rotateAll('X', 0.5); scene.rotateAll('Y', 0.6);
      scene.nodeRadius = entry.slug.startsWith('param') || ['marching_cubes', 'fractal', 'hull'].includes(entry.slug) ? 1.5 : 4;
      draw(setupCanvas(c, 300), scene);
    } catch (e) { c.replaceWith(Object.assign(document.createElement('p'), { className: 'err small', textContent: e.message })); }
  }
});
