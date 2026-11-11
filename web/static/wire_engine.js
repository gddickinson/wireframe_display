// Wireframe engine: a port of pygame_MatrixWireframe.Wireframe and wireframe_scene.WireframeScene (pure maths, no DOM).
// Nodes are rows [x, y, z, 1] in screen pixels (y down). translationMatrix is applied to row vectors (v·M); rotation,
// scale and shear matrices about a centre c as c + M·(v − c), exactly as the Python viewer. Projection is orthographic:
// a node is drawn at its (x, y). Tested against the Python code by wire_engine.test.mjs.

const copyNodes = nodes => nodes.map(n => n.slice());

// --- matrices (wireframe_scene.py) ------------------------------------------------------------------------------------
export const translationMatrix = (dx = 0, dy = 0, dz = 0) => [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [dx, dy, dz, 1]];
export const scaleMatrix = (sx = 0, sy = 0, sz = 0) => [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]];
export const shearMatrix = (hxy = 0, hxz = 0, hyz = 0, hyx = 0, hzx = 0, hzy = 0) =>
  [[1, hxy, hxz, 0], [hyx, 1, hyz, 0], [hzx, hzy, 1, 0], [0, 0, 0, 1]];
export function rotateXMatrix(r) { const c = Math.cos(r), s = Math.sin(r); return [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]; }
export function rotateYMatrix(r) { const c = Math.cos(r), s = Math.sin(r); return [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]; }
export function rotateZMatrix(r) { const c = Math.cos(r), s = Math.sin(r); return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]; }
export const identityMatrix = () => [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
export const perspectiveMatrix = c => [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, -1 / c, 1]];
const MATRICES = { rotateXMatrix, rotateYMatrix, rotateZMatrix };

// c + M·(v − c) for every node (Wireframe.scale / shear / rotate)
function aboutCentre(nodes, c, M) {
  for (const v of nodes) {
    const d = [v[0] - c[0], v[1] - c[1], v[2] - c[2], v[3] - c[3]];
    for (let i = 0; i < 4; i++) v[i] = c[i] + (M[i][0] * d[0] + M[i][1] * d[1] + M[i][2] * d[2] + M[i][3] * d[3]);
  }
}

// --- Wireframe (pygame_MatrixWireframe.py) ----------------------------------------------------------------------------
export class Wireframe {
  constructor(nodes = null) {
    this.nodes = []; this.edges = []; this.faces = [];
    this.initiated = false; this.history = []; this.historyMax = 20;
    this.colour = [255, 255, 200]; this.name = '';
    if (nodes && nodes.length) this.addNodes(nodes);
  }
  addNodes(list, reset = false) {
    if (reset) this.nodes = [];
    for (const p of list) this.nodes.push([+p[0], +p[1], +p[2], 1]);
    if (!this.initiated) { this.history.push(copyNodes(this.nodes)); this.initiated = true; }
  }
  addEdges(list) { for (const [a, b] of list) this.edges.push([a, b]); }
  addFaces(faceList, reset = false) {
    if (reset) { this.faces = []; this.edges = []; }
    for (const f of faceList) {
      if (f.every(n => n < this.nodes.length)) {
        this.faces.push(f.slice());
        this.addEdges(f.map((_, n) => [f[(n - 1 + f.length) % f.length], f[n]]));
      }
    }
  }
  findCentre() {
    const m = [0, 0, 0, 0], n = this.nodes.length;
    for (const v of this.nodes) for (let i = 0; i < 4; i++) m[i] += v[i];
    return m.map(s => s / n);
  }
  transform(M) {
    this.nodes = this.nodes.map(v => [0, 1, 2, 3].map(j => v[0] * M[0][j] + v[1] * M[1][j] + v[2] * M[2][j] + v[3] * M[3][j]));
  }
  scale(c, M) { aboutCentre(this.nodes, c, M); }
  shear(c, M) { aboutCentre(this.nodes, c, M); }
  rotate(c, M) { aboutCentre(this.nodes, c, M); }
  moveTo(x, y, z) {
    const c = this.findCentre(), v = [x - c[0], y - c[1], z - c[2]];
    this.update('transform', translationMatrix(v[0], v[1], v[2]));
  }
  changeShape(wf) {
    const c = this.findCentre();
    this.nodes = copyNodes(wf.nodes); this.faces = wf.faces.map(f => f.slice()); this.edges = wf.edges.map(e => e.slice());
    this.moveTo(c[0], c[1], c[2]);
  }
  update(fn, ...args) {
    this.history.push(copyNodes(this.nodes));
    this[fn](...args);
    if (this.history.length === this.historyMax) this.history.shift();
  }
  reduceHistory() { if (this.history.length > 0) this.history.shift(); }
  // Subdivide each triangle into four, new vertices on the shape's own sphere (centre, mean vertex radius).
  subdivide() {
    const faces = this.faces.map(f => f.slice());
    if (!faces.length || faces.some(f => f.length !== 3)) return;
    const c = this.findCentre();
    const verts = this.nodes.map(v => [v[0] - c[0], v[1] - c[1], v[2] - c[2]]);
    const radius = verts.reduce((s, v) => s + Math.hypot(...v), 0) / verts.length;
    const push = (a, b) => { const s = [a[0] + b[0], a[1] + b[1], a[2] + b[2]], n = Math.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2);
      verts.push(s.map(x => radius * (x / n))); };
    const triangles = faces.length;
    for (let fi = 0; fi < triangles; fi++) {
      const face = faces[fi], [a, b, cc] = face.map(i => verts[i]);
      push(a, b); push(b, cc); push(a, cc);
      const i = verts.length - 3, j = i + 1, k = i + 2;
      faces.push([i, j, k], [face[0], i, k], [i, face[1], j]);
      faces[fi] = [k, j, face[2]];
    }
    this.nodes = []; this.faces = []; this.edges = [];
    this.addNodes(verts.map(v => [v[0] + c[0], v[1] + c[1], v[2] + c[2]]));
    this.addFaces(faces);
  }
}

// Screen positions (orthographic projection): [[x, y], ...]
export const project = wf => wf.nodes.map(v => [v[0], v[1]]);

// --- key table (wireframe_scene.KEY_BINDINGS) --------------------------------------------------------------------------
export const KEY_BINDINGS = [
  ['LEFT', 'translateAll', [[-10, 0, 0]], 'move left'], ['RIGHT', 'translateAll', [[10, 0, 0]], 'move right'],
  ['DOWN', 'translateAll', [[0, 10, 0]], 'move down'], ['UP', 'translateAll', [[0, -10, 0]], 'move up'],
  ['EQUALS', 'scaleAll', [1.05], 'zoom in (scale about the screen centre)'], ['MINUS', 'scaleAll', [0.95], 'zoom out'],
  ['q', 'rotateAll', ['X', 0.1], 'rotate about X +'], ['w', 'rotateAll', ['X', -0.1], 'rotate about X -'],
  ['a', 'rotateAll', ['Y', 0.1], 'rotate about Y +'], ['s', 'rotateAll', ['Y', -0.1], 'rotate about Y -'],
  ['z', 'rotateAll', ['Z', 0.1], 'rotate about Z +'], ['x', 'rotateAll', ['Z', -0.1], 'rotate about Z -'],
  ['e', 'toggleEdges', [], 'show / hide edges'], ['d', 'toggleNodes', [], 'show / hide nodes'],
  ['t', 'shearAll', [[-0.1, 0, 0, 0, 0, 0]], 'shear xy -'], ['y', 'shearAll', [[0.1, 0, 0, 0, 0, 0]], 'shear xy +'],
  ['g', 'shearAll', [[0, -0.1, 0, 0, 0, 0]], 'shear xz -'], ['h', 'shearAll', [[0, 0.1, 0, 0, 0, 0]], 'shear xz +'],
  ['b', 'shearAll', [[0, 0, -0.1, 0, 0, 0]], 'shear yz -'], ['n', 'shearAll', [[0, 0, 0.1, 0, 0, 0]], 'shear yz +'],
  ['m', 'subdivideAll', [], 'subdivide triangles'], ['0', 'centerAll', [], 'centre on screen'],
  ['9', 'changeAllShapes', [], 'change every shape to the tetrahedron'],
];
// browser KeyboardEvent.key -> key name of the table
export const keyName = k => ({ ArrowLeft: 'LEFT', ArrowRight: 'RIGHT', ArrowUp: 'UP', ArrowDown: 'DOWN', '=': 'EQUALS', '+': 'EQUALS', '-': 'MINUS' }[k]
  || (k.length === 1 ? k.toLowerCase() : k));

// --- scene (wireframe_scene.WireframeScene) ------------------------------------------------------------------------------
export class Scene {
  constructor(width, height, { rand = Math.random, tetra = null } = {}) {
    Object.assign(this, { width, height, rand, tetra, wireframes: new Map(), displayNodes: true, displayEdges: true,
      nodeColour: [255, 255, 255], edgeColour: [200, 200, 200], nodeRadius: 4, background: [10, 10, 50],
      animate: false, updateEvent: false, randomEvent: false, showHistory: false, colourShift: false,
      useWireframeColour: false, useRandomColour: false, moveCenter: [width / 2, height / 2, 0], yoyoCounter: 0, history: [] });
  }
  addWireframe(name, wf) { this.wireframes.set(name, wf); this.history.push(name); }
  removeWireframe(name) { this.wireframes.delete(name); this.history = this.history.filter(n => n !== name); }
  each(fn) { for (const wf of this.wireframes.values()) fn(wf); }
  getRandomColour() { return [0, 0, 0].map(() => Math.floor(this.rand() * 256)); }
  shiftColour(c, count) { return [Math.max(c[0] - count, 0), Math.max(c[1] - 2 * count, 0), Math.max(c[2] - 5 * count, 0), 1 / (count + 1)]; }
  shearAll(v) { const M = shearMatrix(v[0], v[1], v[2], v[0], v[1], v[2]); this.each(wf => wf.update('shear', wf.findCentre(), M)); }
  translateAll(v) {
    const M = translationMatrix(v[0], v[1], v[2]);
    this.moveCenter = [this.moveCenter[0] + v[0], this.moveCenter[1] + v[1], this.moveCenter[2] + v[2]];
    this.each(wf => wf.update('transform', M));
  }
  scaleAll(s) { const c = [this.width / 2, this.height / 2, 0, 0], M = scaleMatrix(s, s, s); this.each(wf => wf.update('scale', c, M)); }
  rotateAll(axis, theta) { this.each(wf => { const c = wf.findCentre(); wf.update('rotate', c, MATRICES['rotate' + axis + 'Matrix'](theta)); }); }
  centerAll() { this.moveAll(this.width / 2, this.height / 2, 0); }
  moveAll(x, y, z) {
    this.each(wf => { const c = wf.findCentre(); wf.update('transform', translationMatrix(x - c[0], y - c[1], z - c[2])); });
  }
  subdivideAll() { this.each(wf => wf.subdivide()); }
  toggleEdges() { this.displayEdges = !this.displayEdges; }
  toggleNodes() { this.displayNodes = !this.displayNodes; }
  randomColour() { this.nodeColour = this.getRandomColour(); }
  changeAllShapes(shape = null) { const s = shape || (this.tetra && this.tetra()); if (s) this.each(wf => wf.changeShape(s)); }
  zMotion(direction = 'away') { this.scaleAll(direction === 'away' ? 0.85 : 1.1762); }
  yoyo() { this.yoyoCounter < 20 ? this.zMotion() : this.zMotion('towards'); if (++this.yoyoCounter >= 40) this.yoyoCounter = 0; }
  spin() {
    this.translateAll([-10, 0, 0]); this.rotateAll('Z', 0.1); this.translateAll([10, 0, 0]); this.rotateAll('X', 0.1);
    this.translateAll([0, 10, 0]); this.rotateAll('Y', 0.1); this.translateAll([0, -10, 0]);
  }
  animate_update() { this.spin(); this.yoyo(); }
  update() { this.each(wf => wf.reduceHistory()); }
  random_update() { this.randomColour(); }
  applyKey(key) {
    const b = KEY_BINDINGS.find(k => k[0] === key);
    if (!b) return false;
    this[b[1]](...b[2]);
    return true;
  }
  run(method, args = []) { this[method](...args); }
}
