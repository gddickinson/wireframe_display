// Home page: hero with the desktop app's start-up scene (the octasphere) spinning live, counts, entry cards.
import { route, api, app, esc, list, animate } from './core.js';
import { catalogue, newScene, makeShape, fit, setupCanvas, draw } from './render.js';

const DOORS = [
  ['#/viewer', 'Viewer', 'The desktop viewer in your browser: the same keys, transforms, animation and history trail, plus drag to rotate, wheel to zoom and your own OBJ files.'],
  ['#/shapes', 'Shapes', 'Every demo shape in the code: solids, terrain, parametric surfaces and two computed by SciPy / scikit-image.'],
  ['#/check', 'Cross-check', 'The browser engine against the Python: the same shape and the same key presses must land on the same pixels.'],
  ['#/about', 'About', 'The matrices and the projection, how to run the desktop and web apps, credits.'],
];

route('home', async () => {
  const [info, { shapes }] = await Promise.all([api('/info'), catalogue()]);
  app().innerHTML = `
  <section class="hero two herow">
    <div>
      <p class="kicker">Matrices · projection · wireframes</p>
      <h1 class="glow">Wireframe Display</h1>
      <p class="lead">3D shapes as nodes and edges, moved by 4 × 4 transformation matrices and projected onto a flat screen.
      This is the web version of a Python/Pygame wireframe viewer: the same shapes and the same maths, running in your browser.</p>
      <a class="btn" href="#/viewer">Open the viewer</a><a class="btn alt" href="#/shapes">Browse the shapes</a>
    </div>
    <figure class="fig"><canvas id="hero" role="img" aria-label="The octasphere, the desktop app's start-up shape, turning with the app's own spin animation"></canvas>
      <figcaption class="small mute">Octasphere (pyrr quaternions, 360 nodes), the desktop app's start-up shape, turned by small rotateAll steps every 50 ms.</figcaption></figure>
  </section>
  <div class="tiles">
    <div class="tile"><b>${info.shapes}</b><span>demo shapes</span></div>
    <div class="tile"><b>${info.parametric_functions}</b><span>parametric surfaces</span></div>
    <div class="tile"><b>${info.keys}</b><span>key bindings</span></div>
    <div class="tile"><b>4 × 4</b><span>homogeneous matrices</span></div>
  </div>
  <div class="grid wide">${list(DOORS, ([href, t, d]) => `<a class="card door" href="${href}"><b>${t}</b><span>${d}</span></a>`)}</div>
  <h2>Shapes</h2>
  <p>${list(shapes, s => `<a class="tag ${s.python_only ? 'b' : 'a'}" href="#/viewer?shape=${s.slug}">${esc(s.name)}</a>`)}</p>`;
  hero(shapes.find(s => s.slug === 'octasphere'));
});

async function hero(entry) {
  const canvas = document.getElementById('hero'), ctx = setupCanvas(canvas, 480);
  const scene = newScene();
  scene.addWireframe('o', await makeShape(entry));
  fit(scene, 200);
  scene.nodeRadius = 3;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { scene.rotateAll('X', 0.5); scene.rotateAll('Y', 0.6); draw(ctx, scene); return; }
  let last = null;
  animate(t => {
    if (last === null || t - last >= 50) { // the desktop ANIMATE_EVENT timer is 50 ms
      last = t;
      scene.rotateAll('Z', 0.02); scene.rotateAll('X', 0.02); scene.rotateAll('Y', 0.02);
      draw(ctx, scene);
    }
  });
}
