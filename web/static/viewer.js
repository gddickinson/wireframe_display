// Viewer: the desktop ProjectionViewer in the browser. Same 600 x 600 screen, key table, scene operations, timed
// events (animation every 50 ms, random colour every 50 ms, history shrink every 500 ms) and drawing; plus drag to
// rotate, wheel to zoom, Fit, several shapes at once and OBJ files from your computer.
import { route, app, esc, list, animate, rng, slider } from './core.js';
import { keyName, KEY_BINDINGS as KEYS } from './wire_engine.js';
import { loadOBJ, build } from './shapes_engine.js';
import { catalogue, newScene, makeShape, fit, setupCanvas, draw } from './render.js';

const GROUPS = [['Move', ['LEFT', 'RIGHT', 'UP', 'DOWN']], ['Zoom', ['EQUALS', 'MINUS']], ['Rotate', ['q', 'w', 'a', 's', 'z', 'x']],
  ['Shear', ['t', 'y', 'g', 'h', 'b', 'n']], ['Other', ['e', 'd', 'm', '0', '9']]];
const LABEL = { LEFT: '←', RIGHT: '→', UP: '↑', DOWN: '↓', EQUALS: '=', MINUS: '−' };
const FLAGS = [ // ProjectionViewer attributes that have no key in the desktop app (set in code there)
  ['animate', 'Animate: spin + yo-yo every 50 ms'], ['showHistory', 'History trail (last 19 positions)'],
  ['colourShift', 'Shift trail colour'], ['updateEvent', 'Shorten the trail every 500 ms'],
  ['randomEvent', 'New random node colour every 50 ms'], ['useRandomColour', 'Random colour for every edge and node'],
  ['useWireframeColour', "Nodes in the wireframe's own colour"],
];

route('viewer', async (_args, query) => {
  const { shapes } = await catalogue();
  const groups = [...new Set(shapes.map(s => s.group))];
  let seed = 1, current = shapes.find(s => s.slug === query.shape) || shapes.find(s => s.slug === 'octasphere');
  app().innerHTML = `<p class="kicker">Viewer</p><h1 class="glow">Wireframe viewer</h1>
  <div class="sim">
    <div class="controls">
      <label for="shape">Shape</label>
      <select id="shape">${list(groups, g => `<optgroup label="${esc(g)}">${list(shapes.filter(s => s.group === g),
        s => `<option value="${s.slug}" ${s.slug === current.slug ? 'selected' : ''}>${esc(s.name)}${s.python_only ? ' (Python)' : ''}</option>`)}</optgroup>`)}</select>
      <button id="load">Load</button><button class="alt" id="add" title="Add to the scene next to what is there">Add</button><button class="alt" id="fit">Fit</button>
      <button class="alt" id="reseed" hidden>New landscape</button>
      <label for="obj">Or open an OBJ file</label><input type="file" id="obj" accept=".obj,text/plain">
      <div id="keys"></div>
      <div id="sliders"></div>
      <label class="chk"><input type="checkbox" id="demo"> Timed demo: a new shape every 10 s (desktop: --demo)</label>
      <div class="flags">${list(FLAGS, ([k, t]) => `<label class="chk"><input type="checkbox" data-flag="${k}"> ${esc(t)}</label>`)}</div>
      <div class="readout" id="readout"></div>
      <p class="small mute" id="err"></p>
    </div>
    <div class="view">
      <canvas id="screen" tabindex="0" role="img" aria-label="Wireframe view. Drag to rotate, wheel to zoom, keys as in the desktop app."></canvas>
      <p class="small mute">Drag to rotate · wheel or = / − to zoom · click the view, then use the desktop keys (Q W A S Z X rotate,
      arrows move, T Y G H B N shear, E / D edges / nodes, M subdivide, 0 centre, 9 tetrahedron).</p>
      <p class="small mute" id="about"></p>
    </div>
  </div>`;

  const canvas = document.getElementById('screen'), ctx = setupCanvas(canvas);
  let scene = newScene(rng(99)), dirty = true;
  const touch = () => { dirty = true; };
  const err = msg => { document.getElementById('err').textContent = msg || ''; };

  async function load(entry, add = false) {
    err(); scene.stopTimedDemo(); document.getElementById('demo').checked = false;
    try {
      const wf = await makeShape(entry, rng(seed));
      if (!add) scene.wireframes = new Map();
      let name = entry.slug, k = 2;
      while (scene.wireframes.has(name)) name = `${entry.slug}-${k++}`;
      scene.addWireframe(name, wf);
      if (add) scene.centerAll(); else fit(scene);
      current = entry;
      document.getElementById('reseed').hidden = entry.fn !== 'FractalLandscape';
      document.getElementById('about').innerHTML = `<b>${esc(entry.name)}</b>: <code>${esc(entry.fn)}(${esc(fmtArgs(entry.args))})</code>
        ${entry.doc ? '· ' + esc(entry.doc) : ''}${entry.python_only ? ' · computed by the Python code (SciPy / scikit-image) and loaded from the API' : ''}`;
      history.replaceState(null, '', `#/viewer?shape=${entry.slug}`);
    } catch (e) { err(e.message); }
    touch();
  }
  const pick = () => shapes.find(s => s.slug === document.getElementById('shape').value);
  document.getElementById('load').onclick = () => load(pick());
  document.getElementById('add').onclick = () => load(pick(), true);
  document.getElementById('fit').onclick = () => { fit(scene); touch(); };
  document.getElementById('reseed').onclick = () => { seed++; load(current); };
  document.getElementById('obj').onchange = async e => {
    const file = e.target.files[0]; if (!file) return;
    try {
      const wf = loadOBJ(await file.text()); scene.wireframes = new Map(); scene.addWireframe(file.name, wf); fit(scene); err();
      document.getElementById('about').textContent = `${file.name}: parsed in your browser (port of loadOBJ); nothing is uploaded.`;
    } catch (ex) { err(`${file.name}: ${ex.message}`); }
    touch();
  };

  // key buttons, built from the key table
  const keysHost = document.getElementById('keys');
  keysHost.innerHTML = list(GROUPS, ([g, ks]) => `<div class="kgroup"><span class="small mute">${g}</span><div class="seg">${list(ks, k => {
    const b = keyRow(k);
    return `<button data-key="${k}" title="${esc(b[3])} (key ${esc(LABEL[k] || k.toUpperCase())})">${esc(LABEL[k] || k.toUpperCase())}</button>`;
  })}</div></div>`);
  keysHost.onclick = e => { const k = e.target.closest('[data-key]'); if (k) { scene.applyKey(k.dataset.key); touch(); } };
  slider(document.getElementById('sliders'), { id: 'radius', label: 'Node radius', min: 0, max: 10, value: 4, unit: ' px' },
    v => { scene.nodeRadius = v; touch(); });
  document.getElementById('demo').onchange = e => {
    if (e.target.checked) scene.startTimedDemo(shapes.filter(s => !s.python_only).map(s => [s.slug, () => build(s, { rand: rng(seed) })]), 10);
    else scene.stopTimedDemo();
    touch();
  };
  document.querySelectorAll('[data-flag]').forEach(cb => { cb.onchange = () => { scene[cb.dataset.flag] = cb.checked; touch(); }; });

  canvas.addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (scene.applyKey(keyName(e.key))) { e.preventDefault(); touch(); }
  });
  let last = null;
  canvas.addEventListener('pointerdown', e => { last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); canvas.focus(); });
  canvas.addEventListener('pointermove', e => {
    if (!last) return;
    const k = 600 / canvas.clientWidth, dx = (e.clientX - last[0]) * k, dy = (e.clientY - last[1]) * k;
    last = [e.clientX, e.clientY];
    if (dx) scene.rotateAll('Y', dx * 0.01);
    if (dy) scene.rotateAll('X', -dy * 0.01);
    touch();
  });
  const up = () => { last = null; };
  canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('wheel', e => { e.preventDefault(); scene.scaleAll(e.deltaY < 0 ? 1.05 : 0.95); touch(); }, { passive: false });

  // timed events as in ProjectionViewer.run(): ANIMATE and RANDOM every 50 ms, UPDATE every 500 ms
  let t50 = null, t500 = null;
  animate(t => {
    if (t50 === null) { t50 = t; t500 = t; }
    while (t - t50 >= 50) {
      t50 += 50;
      if (scene.animate) { scene.animate_update(); dirty = true; }
      if (scene.randomEvent) { scene.random_update(); dirty = true; }
    }
    while (t - t500 >= 500) { t500 += 500; if (scene.updateEvent) { scene.update(); dirty = true; } }
    if (scene.useRandomColour) dirty = true;
    if (scene.lifetime !== null && scene.timedAction(Date.now()).length) dirty = true;
    if (!dirty) return;
    dirty = false;
    draw(ctx, scene);
    readout(scene);
  });
  await load(current);
});

function keyRow(k) { return KEYS.find(r => r[0] === k); }

function fmtArgs(a) {
  return Object.entries(a).map(([k, v]) => `${k}=${Array.isArray(v) ? `(${v.join(', ')})` : typeof v === 'string' ? v : v}`).join(', ');
}

function readout(scene) {
  let nodes = 0, edges = 0, faces = 0, hist = 0;
  for (const wf of scene.wireframes.values()) { nodes += wf.nodes.length; edges += wf.edges.length; faces += wf.faces.length; hist = Math.max(hist, wf.history.length); }
  const first = scene.wireframes.values().next().value, c = first ? first.findCentre() : [0, 0, 0];
  const row = (k, v) => `<div><span>${k}</span><b>${v}</b></div>`;
  document.getElementById('readout').innerHTML = row('shapes', [...scene.wireframes.keys()].join(', ').slice(0, 28) || 0) + row('nodes', nodes.toLocaleString('en')) +
    row('edges', edges.toLocaleString('en')) + row('faces', faces.toLocaleString('en')) + row('history', hist) +
    row('centre', c.slice(0, 3).map(v => v.toFixed(0)).join(', ')) +
    row('edges / nodes', `${scene.displayEdges ? 'on' : 'off'} / ${scene.displayNodes ? 'on' : 'off'}`);
}
