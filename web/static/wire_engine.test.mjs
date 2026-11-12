// Node test: the JS engine reproduces the Python numbers (pygame_basicShapes, pygame_parametricShapes,
// pygame_MatrixWireframe, wireframe_scene). Reference: tests/fixtures/engine_fixture.json, written by
// `python scripts/make_engine_fixture.py --write`; tests/test_web.py checks that file is current.
// Run: node web/static/wire_engine.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as E from './wire_engine.js';
import { build, loadOBJ, Octahedron, Octasphere } from './shapes_engine.js';

const FIX = JSON.parse(readFileSync(fileURLToPath(new URL('../../tests/fixtures/engine_fixture.json', import.meta.url)), 'utf8'));
const detRand = () => { let k = 0; return () => (++k * 0.6180339887498949) % 1.0; };
let n = 0, worst = 0;
// relative tolerance; float32 shapes (octasphere quaternions) are compared at float32 precision
function close(a, b, tol, what) {
  const d = Math.abs(a - b) / Math.max(1, Math.abs(b));
  worst = Math.max(worst, tol < 1e-6 ? d : 0);
  assert.ok(d <= tol, `${what}: js ${a} vs py ${b}`);
}
function sameSample(nodes, s, tol, what, cols = 3) {
  assert.equal(nodes.length, s.n, `${what}: node count`);
  s.idx.forEach((i, r) => s.rows[r].forEach((v, c) => close(nodes[i][c], v, tol, `${what} node ${i}[${c}]`)));
  for (let c = 0; c < cols; c++) close(nodes.reduce((t, v) => t + v[c], 0), s.sums[c], tol, `${what} column ${c} sum`);
}
const tolFor = slug => (slug.startsWith('octasphere') ? 1e-6 : 1e-9);

// 1. matrices
for (const [name, args, M] of FIX.matrices) {
  E[name](...args).forEach((row, i) => row.forEach((v, j) => close(v, M[i][j], 1e-12, `${name}[${i}][${j}]`))); n++;
}
// 2. every ported catalogue shape, built from the same (fn, args) spec
for (const [slug, c] of Object.entries(FIX.shapes)) {
  const wf = build(c.spec, { rand: detRand() });
  sameSample(wf.nodes, c.nodes, tolFor(slug), slug);
  assert.equal(wf.edges.length, c.edges, `${slug} edges`); assert.equal(wf.faces.length, c.faces, `${slug} faces`);
  assert.deepEqual(wf.edges.slice(0, 12), c.edges_head, `${slug} first edges`); n++;
}
// 3. scene runs: the same key / animation operations, step by step (4D nodes, history length, move centre, yoyo)
for (const [slug, c] of Object.entries(FIX.scenes)) {
  const scene = new E.Scene(600, 600);
  scene.addWireframe(slug, build(FIX.shapes[slug].spec, { rand: detRand() }));
  FIX.ops.forEach(([method, args], k) => {
    scene.run(method, args);
    const wf = scene.wireframes.get(slug), st = c.steps[k];
    sameSample(wf.nodes, st.nodes, tolFor(slug) * 10, `${slug} after ${k}:${method}`, 4);
    assert.equal(wf.history.length, st.history, `${slug} history after ${method}`);
  });
  c.move_center.forEach((v, i) => close(scene.moveCenter[i], v, 1e-12, `${slug} moveCenter`));
  assert.equal(scene.yoyoCounter, c.yoyo); n++;
}
// 4. subdivision onto the shape's own sphere
{
  const o = Octahedron(); o.subdivide();
  sameSample(o.nodes, FIX.subdivide.octa.nodes, 1e-9, 'subdivided octahedron'); assert.equal(o.faces.length, FIX.subdivide.octa.faces);
  const s = Octasphere(); s.moveTo(300, 300, 0); s.subdivide();
  sameSample(s.nodes, FIX.subdivide.octasphere_centred.nodes, 1e-6, 'subdivided octasphere'); n += 2;
}
// 5. OBJ parser on tests/fixtures/sample.obj
{
  const m = loadOBJ(readFileSync(fileURLToPath(new URL('../../tests/fixtures/sample.obj', import.meta.url)), 'utf8'));
  assert.deepEqual(m.nodes.map(v => v.slice(0, 3)), FIX.obj.nodes);
  assert.deepEqual(m.faces, FIX.obj.faces); assert.deepEqual(m.edges, FIX.obj.edges); n++;
}
// 5b. fitAll and the timed demo (pygame_MatrixDisplay.py --demo)
for (const [slug, smp] of Object.entries(FIX.fit)) {
  const sc = new E.Scene(600, 600); sc.addWireframe(slug, build(FIX.shapes[slug].spec)); sc.rotateAll('X', 0.4); sc.fitAll();
  sameSample(sc.wireframes.get(slug).nodes, smp, 1e-9, `fitAll ${slug}`, 4); n++;
}
{
  const sc = new E.Scene(600, 600), t0 = Date.now();
  assert.deepEqual(sc.timedAction(t0 + 1e6), []);
  sc.startTimedDemo([['cube', () => build(FIX.shapes.cube.spec)], ['octa', () => build(FIX.shapes.octa.spec)]], 5);
  assert.deepEqual([...sc.wireframes.keys()], ['cube']);
  assert.deepEqual(sc.timedAction(Date.now() + 1000), []);
  assert.deepEqual(sc.timedAction(Date.now() + 6000), ['cube']); assert.deepEqual([...sc.wireframes.keys()], ['octa']);
  assert.deepEqual(sc.timedAction(Date.now() + 6000), ['octa']); assert.deepEqual([...sc.wireframes.keys()], ['cube']);
  sc.stopTimedDemo(); assert.deepEqual(sc.timedAction(Date.now() + 1e6), []); n++;
}
// 6. key table identical to wireframe_scene.KEY_BINDINGS
assert.deepEqual(E.KEY_BINDINGS, FIX.keys); n++;
// 7. keyboard mapping and the orthographic projection
assert.equal(E.keyName('ArrowLeft'), 'LEFT'); assert.equal(E.keyName('Q'), 'q'); assert.equal(E.keyName('='), 'EQUALS');
const sc = new E.Scene(600, 600); const cube = build(FIX.shapes.cube.spec); sc.addWireframe('c', cube);
sc.applyKey('LEFT'); assert.deepEqual(E.project(cube)[0], [290, 300]); n++;

console.log(`wire_engine: ${n} checks passed (${Object.keys(FIX.shapes).length} shapes, ${Object.keys(FIX.scenes).length} scene runs ` +
  `x ${FIX.ops.length} ops; worst relative difference on float64 shapes ${worst.toExponential(1)})`);
