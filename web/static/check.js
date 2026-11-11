// Cross-check: for every shape, the Python scene (wireframe_scene.py, via the API) and the JS engine run the same
// operations from the same start; the table shows how far apart the final screen positions are.
import { route, app, api, esc, list, isStatic } from './core.js';
import { catalogue, newScene, makeShape } from './render.js';

const detRand = () => { let k = 0; return () => (++k * 0.6180339887498949) % 1.0; };

route('check', async () => {
  const { shapes, check_exported } = await catalogue();
  app().innerHTML = `<p class="kicker">Cross-check</p><h1 class="glow">Browser vs Python</h1>
  <p class="lead">Each shape is built twice, by the Python constructors and by the JS port, put on a 600 × 600 screen and put through
  the same operations: the keys 0, Q, A, Z, =, T, H, →, three animation ticks (spin + yo-yo), M (subdivide) and 0 again.
  The Python result ${isStatic() ? 'was computed when this site was built' : 'is computed now by the local server'}; the JS result
  is computed in your browser. Agreement is measured in screen pixels.</p>
  <p class="small mute" id="ops"></p>
  <div class="scroll"><table class="check"><thead><tr><th>Shape</th><th>Nodes (Py / JS)</th><th>Edges (Py / JS)</th>
    <th>History</th><th>Max |Δ| px</th><th></th></tr></thead><tbody id="rows"></tbody></table></div>
  <p class="small mute">The fractal landscape normally uses random heights; here both sides draw them from the same sequence
  frac(k × 0.618…), k = 1, 2, …. The convex hull uses NumPy seed 0. Octasphere vertices are float32 in the Python (pyrr), so they agree
  to float32 precision. The Python-only shapes start from the nodes the Python built.</p>
  <p id="summary"></p>`;
  const rows = document.getElementById('rows');
  let ok = 0, done = 0;
  for (const s of shapes) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><a href="#/viewer?shape=${s.slug}">${esc(s.name)}</a></td><td colspan="5" class="mute">checking…</td>`;
    rows.appendChild(tr);
    try {
      if (isStatic() && !check_exported.includes(s.slug)) throw new Error('Checked in the local app only (the subdivided mesh is too large to export).');
      const py = await api('/check/' + s.slug);
      if (!document.getElementById('ops').textContent)
        document.getElementById('ops').textContent = 'Operations: ' + py.ops.map(([m, a]) => `${m}(${a.map(x => JSON.stringify(x)).join(', ')})`).join(' → ');
      const scene = newScene();
      const wf = await makeShape(s, detRand());
      scene.addWireframe(s.slug, wf);
      for (const [m, a] of py.ops) scene.run(m, a);
      const w = scene.wireframes.get(s.slug);
      let d = 0;
      if (w.nodes.length === py.points.length) w.nodes.forEach((v, i) => { d = Math.max(d, Math.abs(v[0] - py.points[i][0]), Math.abs(v[1] - py.points[i][1])); });
      const pass = w.nodes.length === py.nodes && w.edges.length === py.edges && w.history.length === py.history && d < 1e-3;
      ok += pass; done++;
      tr.innerHTML = `<td><a href="#/viewer?shape=${s.slug}">${esc(s.name)}</a></td><td class="num">${py.nodes} / ${w.nodes.length}</td>
        <td class="num">${py.edges} / ${w.edges.length}</td><td class="num">${py.history} / ${w.history.length}</td>
        <td class="num">${w.nodes.length === py.points.length ? d.toExponential(1) : '—'}</td><td class="${pass ? 'ok' : 'err'}">${pass ? 'match' : 'differs'}</td>`;
    } catch (e) {
      tr.innerHTML = `<td>${esc(s.name)}</td><td colspan="5" class="mute">${esc(e.message)}</td>`;
    }
  }
  document.getElementById('summary').innerHTML = `<b class="${ok === done ? 'ok' : 'err'}">${ok} of ${done} shapes match</b> (|Δ| &lt; 0.001 px, same node, edge and history counts).`;
});
