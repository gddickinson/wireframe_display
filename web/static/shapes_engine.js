// Shape constructors: a port of pygame_basicShapes.py and pygame_parametricShapes.py (pure maths, no DOM).
// build({fn, args}) takes the same (constructor, keyword arguments) spec as shape_catalogue.py on the Python side.
// Where the Python stores float32 (np.float32 arrays, pyrr quaternions) the values are rounded with Math.fround.
// marchingCubes and convexHull need scikit-image / SciPy and are not ported (the viewer loads them from the API).
import { Wireframe } from './wire_engine.js';

const { sin, cos, sqrt, exp, abs, PI: pi, fround: f32 } = Math;
const sign = x => (x > 0 ? 1 : x < 0 ? -1 : 0);

export function Cuboid(args1 = [0, 0, 0], args2 = [0, 0, 0]) {
  const [x, y, z] = args1, [w, h, d] = args2, nodes = [];
  for (const nx of [x, x + w]) for (const ny of [y, y + h]) for (const nz of [z, z + d]) nodes.push([nx, ny, nz]);
  const c = new Wireframe(); c.addNodes(nodes);
  c.addFaces([[0, 1, 3, 2], [7, 5, 4, 6], [4, 5, 1, 0], [2, 3, 7, 6], [0, 2, 6, 4], [5, 7, 3, 1]]);
  c.name = 'cube'; return c;
}

export function Tetrahedron(args1 = [0, 0, 0], edgeLength = 1) {
  const [x, y, z] = args1, e = edgeLength;
  const t = new Wireframe();
  t.addNodes([[e + x, e + y, e + z], [e + x, -e + y, -e + z], [-e + x, e + y, -e + z], [-e + x, -e + y, e + z]]);
  t.addFaces([[0, 1, 2], [2, 3, 0], [1, 3, 2], [3, 1, 0]]);
  t.name = 'tetra'; return t;
}

export function Octahedron() {
  const f = f32(sqrt(2.0) / 2.0);
  const o = new Wireframe();
  o.addNodes([[0, -1, 0], [-f, 0, f], [f, 0, f], [f, 0, -f], [-f, 0, -f], [0, 1, 0]]);
  o.addFaces([[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4], [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1]]);
  return o;
}

export function Spheroid(args1 = [0, 0, 0], args2 = [0, 0, 0], resolution = 10) {
  const [x, y, z] = args1, [rx, ry, rz] = args2, r = resolution, s = new Wireframe();
  const lat = [], lon = [], nodes = [];
  for (let n = 1; n < r; n++) lat.push(n * pi / r);
  for (let n = 0; n < r; n++) lon.push(n * 2 * pi / r);
  for (const m of lat) for (const n of lon) nodes.push([x + rx * sin(n) * sin(m), y - ry * cos(m), z - rz * cos(n) * sin(m)]);
  s.addNodes(nodes);
  const num = r * (r - 1), faces = [];
  for (let n = 0; n < r; n++) for (let m = 0; m < num - r; m += r)
    faces.push([m + n, (m + r) % num + n, (m + r) % (r * r) + (n + 1) % r, m + (n + 1) % r]);
  s.addFaces(faces);
  s.addNodes([[x, y + ry, z], [x, y - ry, z]]);
  s.addFaces([...Array(r).keys()].map(n => [n, (n + 1) % r, num + 1]));
  const start = num - r;
  s.addFaces([...Array(r).keys()].map(n => [num, start + (n + 1) % r, start + n]));
  s.name = 'spheroid'; return s;
}

export function HorizontalGrid(args1 = [0, 0, 0], args2 = [0, 0], args3 = [0, 0]) {
  const [x, y, z] = args1, [dx, dz] = args2, [nx, nz] = args3, g = new Wireframe(), nodes = [], edges = [];
  for (let n1 = 0; n1 <= nx; n1++) for (let n2 = 0; n2 <= nz; n2++) nodes.push([x + n1 * dx, y, z + n2 * dz]);
  g.addNodes(nodes);
  for (let n1 = 0; n1 <= nx; n1++) for (let n2 = 0; n2 < nz; n2++) edges.push([n1 * (nz + 1) + n2, n1 * (nz + 1) + n2 + 1]);
  for (let n1 = 0; n1 < nx; n1++) for (let n2 = 0; n2 <= nz; n2++) edges.push([n1 * (nz + 1) + n2, (n1 + 1) * (nz + 1) + n2]);
  g.addEdges(edges); g.name = 'grid'; return g;
}

// rand() replaces Python's random.random() (seeded in the browser so a landscape can be reproduced)
export function FractalLandscape(origin = [0, 0, 0], dimensions = [400, 400], iterations = 4, height = 40, rand = Math.random) {
  const midpoint = ns => { const m = 1.0 / ns.length; return [0, 1, 2].map(i => m * ns.reduce((s, n) => s + n[i], 0)); };
  const [x, y, z] = origin, [dx, dz] = dimensions;
  let nodes = [[x, y, z], [x + dx, y, z], [x + dx, y, z + dz], [x, y, z + dz]];
  let edges = [[0, 1], [1, 2], [2, 3], [3, 0]], size = 2;
  for (let i = 0; i < iterations; i++) {
    for (const [n1, n2] of edges) nodes.push(midpoint([nodes[n1], nodes[n2]]));
    const squares = [];
    for (let yy = 0; yy < size - 1; yy++) for (let xx = 0; xx < size - 1; xx++)
      squares.push([xx + yy * size, xx + yy * size + 1, xx + (yy + 1) * size + 1, xx + (yy + 1) * size]);
    for (const q of squares) nodes.push(midpoint(q.map(k => nodes[k])));
    nodes.sort((a, b) => (a[2] - b[2]) || (a[0] - b[0]));
    size = size * 2 - 1;
    edges = [];
    for (let yy = 0; yy < size; yy++) for (let xx = 0; xx < size - 1; xx++) edges.push([xx + yy * size, xx + yy * size + 1]);
    for (let xx = 0; xx < size; xx++) for (let yy = 0; yy < size - 1; yy++) edges.push([xx + yy * size, xx + (yy + 1) * size]);
    const scale = height / 2 ** (i * 0.8);
    for (const node of nodes) node[1] += (rand() - 0.5) * scale;
  }
  const g = new Wireframe(nodes); g.addEdges(edges); return g;
}

export function letter() {
  const P = [[0, 0, 0], [4, 0, 0], [0, 0, 2], [4, 0, 2], [14, 0, 0], [18, 0, 0], [14, 0, 2], [18, 0, 2], [5, 6, 0], [13, 6, 0],
    [5, 6, 2], [13, 6, 2], [7, 13, 0], [11, 13, 0], [7, 13, 2], [11, 13, 2], [9, 17, 0], [9, 17, 2], [7, 24, 0], [11, 24, 0],
    [7, 24, 2], [11, 24, 2]];
  const l = new Wireframe(); l.addNodes(P);
  l.addFaces([[0, 1, 3, 2], [4, 5, 7, 6], [8, 9, 11, 10], [12, 13, 15, 14], [18, 19, 21, 20], [0, 2, 20, 18], [5, 7, 21, 19],
    [1, 3, 10, 8], [4, 6, 11, 9], [12, 14, 17, 16], [13, 15, 17, 16]]);
  return l;
}

export function Icosahedron() {
  const verts = [[0.000, 0.000, 1.000], [0.894, 0.000, 0.447], [0.276, 0.851, 0.447], [-0.724, 0.526, 0.447],
    [-0.724, -0.526, 0.447], [0.276, -0.851, 0.447], [0.724, 0.526, -0.447], [-0.276, 0.851, -0.447],
    [-0.894, 0.000, -0.447], [-0.276, -0.851, -0.447], [0.724, -0.526, -0.447], [0.000, 0.000, -1.000]];
  const faces = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1], [11, 7, 6], [11, 8, 7], [11, 9, 8], [11, 10, 9],
    [11, 6, 10], [1, 6, 2], [2, 7, 3], [3, 8, 4], [4, 9, 5], [5, 10, 1], [6, 7, 2], [7, 8, 3], [8, 9, 4], [9, 10, 5], [10, 6, 1]];
  const ic = new Wireframe(); ic.addNodes(verts); ic.addFaces(faces); return ic;
}

// --- parametric surfaces (pygame_parametricShapes.py) -----------------------------------------------------------------
export const SURFACES = {
  sphere: (u, v) => [sin(u) * cos(v), cos(u), -sin(u) * sin(v)],
  funnel: (u, v) => [u, u * sin(v), u * cos(v)],
  noseCone: (u, v) => [u * u, u * sin(v), u * cos(v)],
  helix_3D: (u, v) => [(3 + cos(u)) * cos(v), (3 + cos(u)) * sin(v), sin(u) + 0.5 * v],
  klein(u, v) {
    u = u * 2; let x, z;
    if (u < pi) {
      x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(u) * cos(v);
      z = -8 * sin(u) - 2 * (1 - cos(u) / 2) * sin(u) * cos(v);
    } else {
      x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(v + pi);
      z = -8 * sin(u);
    }
    return [x, -2 * (1 - cos(u) / 2) * sin(v), z];
  },
  mobius_tube(u, v) {
    const R = 1.5, n = 3; u = u * 2;
    const p = (a) => Math.pow(abs(a), 2 / n) * sign(a);
    const r = 1.0 * R + 0.125 * sin(u / 2) * p(sin(v)) + 0.5 * cos(u / 2) * p(cos(v));
    return [r * cos(u), r * sin(u), -0.5 * sin(u / 2) * p(cos(v)) + 0.125 * cos(u / 2) * p(sin(v))];
  },
  torus: (u, v) => { const c = 2, a = 1; return [(c + a * cos(u)) * cos(v), (c + a * cos(u)) * sin(v), a * sin(u)]; },
  helix: (u, v) => [u * cos(v), u * sin(v), 3 * v],
  sineSurface: (u, v) => [5 * sin(u), 5 * sin(v), 5 * sin(u + v)],
  eightSurface: (u, v) => [cos(u) * sin(2 * v), sin(u) * sin(2 * v), sin(v)],
  ellipsoid: (u, v) => [5 * cos(u) * sin(v), 5 * sin(u) * sin(v), 20 * cos(v)],
  hyperOctahedron: (u, v) => [(cos(u) * cos(v)) ** 3, (sin(u) * cos(v)) ** 3, sin(v) ** 3],
  geodesic: (u, v) => [sqrt(u) * cos(v), sqrt(u) * sin(v), u],
  cornucopia: (u, v) => [exp(v) * cos(v + exp(v)) * cos(u) * cos(v), exp(v) * sin(v + exp(v)) * cos(u) * sin(v), exp(v) * sin(u)],
};

export function parametric_surface(slices, stacks, func) {
  const f = typeof func === 'string' ? SURFACES[func] : func, verts = [], faces = [];
  for (let i = 0; i <= slices; i++) {
    const theta = i * pi / slices;
    for (let j = 0; j < stacks; j++) verts.push(f(theta, j * 2.0 * pi / stacks).map(f32));
  }
  let v = 0;
  for (let i = 0; i < slices; i++) {
    for (let j = 0; j < stacks; j++) { const nx = (j + 1) % stacks; faces.push([v + j, v + j + stacks, v + nx + stacks, v + nx]); }
    v += stacks;
  }
  const p = new Wireframe(); p.addNodes(verts); p.addFaces(faces); return p;
}

// --- octasphere (get_octasphere with pyrr quaternions) ------------------------------------------------------------------
const qcross = (a, b) => [a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0], -a[0] * b[2] + a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
  a[0] * b[1] - a[1] * b[0] + a[2] * b[3] + a[3] * b[2], -a[0] * b[0] - a[1] * b[1] - a[2] * b[2] + a[3] * b[3]];
const qapply = (q, v) => qcross(q, qcross([v[0], v[1], v[2], 0], [-q[0], -q[1], -q[2], q[3]])).slice(0, 3);
function axisQuat(axis, theta) {
  const n = sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2);
  if (abs(n - 1) > 1e-8 + 1e-5) axis = axis.map(a => a / n); // np.isclose(norm, 1) as in pyrr
  const s = sin(theta * 0.5);
  return [s * axis[0], s * axis[1], s * axis[2], cos(theta * 0.5)];
}
// float32 quaternion from Euler angles [roll, pitch, yaw] (pyrr.quaternion.create_from_eulers on a float32 array)
function eulerQuat32(e) {
  const h = e.map(a => f32(a * 0.5)), m = (...xs) => xs.reduce((p, x) => f32(p * x));
  const [sR, sP, sY] = h.map(a => f32(sin(a))), [cR, cP, cY] = h.map(a => f32(cos(a)));
  return [f32(m(sR, cP, cY) + m(cR, sP, sY)), f32(m(cR, sP, cY) - m(sR, cP, sY)), f32(m(cR, cP, sY) + m(sR, sP, cY)), f32(m(cR, cP, cY) - m(sR, sP, sY))];
}
function geodesic(dst, a, b, segs) {
  const ang = Math.acos(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
  const axis = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  dst.push(a.slice());
  if (segs === 0) return;
  const dt = ang / segs;
  for (let k = 1; k < segs; k++) dst.push(qapply(axisQuat(axis, k * dt), a));
  dst.push(b.slice());
}

export function get_octasphere(ndivisions, radius) { // sphere case only (width/height/depth <= 2 * radius)
  const n = 2 ** ndivisions + 1, patch = [];
  for (let i = 0; i < n; i++) {
    const t = pi * 0.5 * i / (n - 1);
    geodesic(patch, [0, sin(t), cos(t)], [cos(t), sin(t), 0], n - 1 - i);
  }
  const verts = patch.map(v => v.map(c => c * radius)), faces = [];
  let j0 = 0;
  for (let col = 0; col < n - 1; col++) {
    const h = n - 1 - col, j1 = j0 + 1, j2 = j0 + h + 1, j3 = j0 + h + 2;
    for (let row = 0; row < h - 1; row++) faces.push([j0 + row, j1 + row, j2 + row], [j2 + row, j1 + row, j3 + row]);
    faces.push([j0 + h - 1, j1 + h - 1, j2 + h - 1]);
    j0 = j2;
  }
  const half = f32(f32(pi) * 0.5);
  const eulers = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3]].map(e => e.map(k => f32(k * half)));
  const allV = [], allF = [];
  eulers.forEach((e, p) => {
    const q = eulerQuat32(e);
    for (const v of verts) allV.push(qapply(q, v).map(f32));
    for (const f of faces) allF.push(f.map(i => i + p * verts.length));
  });
  return [allV, allF];
}

export function Octasphere(w = 10, h = 10, d = 10) {
  if (Math.max(w, h, d) > 40) throw new Error('Only the sphere case of get_octasphere is ported');
  const [v, f] = get_octasphere(3, 20.0);
  const o = new Wireframe(); o.addNodes(v); o.addFaces(f); return o;
}

// --- OBJ files (pygame_basicShapes.loadOBJ: vertices and faces split into consecutive triangles) -------------------------
export function loadOBJ(text) {
  const vertices = [], triangles = [];
  for (const line of text.split('\n')) {
    const c = line.trim().split(/\s+/);
    if (c[0] === 'f') {
      const idx = c.slice(1).map(s => parseInt(s.split('/')[0], 10) - 1);
      for (let i = 0; i < idx.length - 2; i++) triangles.push(idx.slice(i, i + 3));
    } else if (c[0] === 'v') vertices.push(c.slice(1, 4).map(Number));
  }
  if (!vertices.length) throw new Error('No vertices ("v x y z" lines) found');
  if (triangles.flat().some(i => !(i >= 0 && i < vertices.length))) throw new Error('Face index out of range (negative/relative OBJ indices are not supported)');
  const m = new Wireframe(); m.addNodes(vertices); m.addFaces(triangles); return m;
}

// --- catalogue spec -> wireframe -----------------------------------------------------------------------------------------
const CTORS = { Cuboid, Tetrahedron, Octahedron, Octasphere, Icosahedron, Spheroid, letter, HorizontalGrid, FractalLandscape };
export const PORTED = new Set([...Object.keys(CTORS), 'parametric_surface']);

export function build({ fn, args = {} }, { rand = Math.random } = {}) {
  const a = args;
  switch (fn) {
    case 'Cuboid': return Cuboid(a.args1, a.args2);
    case 'Tetrahedron': return Tetrahedron(a.args1, a.edgeLength);
    case 'Spheroid': return Spheroid(a.args1, a.args2, a.resolution);
    case 'HorizontalGrid': return HorizontalGrid(a.args1, a.args2, a.args3);
    case 'FractalLandscape': return FractalLandscape(a.origin, a.dimensions, a.iterations, a.height, rand);
    case 'parametric_surface': return parametric_surface(a.slices, a.stacks, a.func);
    default:
      if (CTORS[fn]) return CTORS[fn]();
      throw new Error(`${fn} is computed by the Python app (SciPy / scikit-image)`);
  }
}

// A wireframe from nodes/edges/faces sent by the API (python_only shapes)
export function fromData({ nodes, edges = [], faces = [] }) {
  const w = new Wireframe(); w.addNodes(nodes); w.addEdges(edges); w.faces = faces.map(f => f.slice()); return w;
}
