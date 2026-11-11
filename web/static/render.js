// Canvas drawing and shape loading shared by the pages. draw() is a port of ProjectionViewer.display()
// (pygame_MatrixDisplay.py): background, edges as 1 px anti-aliased lines, nodes as filled circles, optional history trail.
import { api } from './core.js';
import { Scene } from './wire_engine.js';
import { build, fromData, Tetrahedron } from './shapes_engine.js';

export const SIZE = 600; // the desktop window is 600 x 600 (pygame_MatrixDisplay.py)
const rgb = c => `rgb(${c[0]},${c[1]},${c[2]})`;

export const catalogue = () => api('/shapes');

// A scene the size of the desktop window; key 9 changes shapes to the same tetrahedron as the desktop default.
export function newScene(rand = Math.random) {
  return new Scene(SIZE, SIZE, { rand, tetra: () => Tetrahedron([0, 0, 0], 100) });
}

// Build a catalogue entry: in the browser, or (SciPy / scikit-image shapes) from the Python nodes served by the API.
export async function makeShape(entry, rand = Math.random) {
  const wf = entry.python_only ? fromData(await api('/shape/' + entry.slug)) : build(entry, { rand });
  wf.name = entry.slug;
  return wf;
}

// Web addition: centre the scene (key 0) and scale it about the screen centre (as the = / - keys do) so the largest
// node distance from the centre is `radius` px. The desktop app only centres; its unit-sized shapes start 1 px wide.
export function fit(scene, radius = 230) {
  scene.centerAll();
  let r = 0;
  for (const wf of scene.wireframes.values()) {
    const c = wf.findCentre();
    for (const v of wf.nodes) r = Math.max(r, Math.hypot(v[0] - c[0], v[1] - c[1], v[2] - c[2]));
  }
  if (r > 0) scene.scaleAll(radius / r);
  for (const wf of scene.wireframes.values()) wf.history = [];
}

export function setupCanvas(canvas, size = SIZE) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = size * dpr; canvas.height = size * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr * size / SIZE, 0, 0, dpr * size / SIZE, 0, 0);
  return ctx;
}

export function draw(ctx, scene, { nodeRadius = scene.nodeRadius } = {}) {
  ctx.fillStyle = rgb(scene.background);
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.lineWidth = 1;
  for (const wf of scene.wireframes.values()) {
    const N = wf.nodes;
    if (scene.displayEdges) {
      if (scene.useRandomColour) {
        for (const [a, b] of wf.edges) {
          ctx.strokeStyle = rgb(scene.getRandomColour());
          ctx.beginPath(); ctx.moveTo(N[a][0], N[a][1]); ctx.lineTo(N[b][0], N[b][1]); ctx.stroke();
        }
      } else {
        ctx.strokeStyle = rgb(scene.edgeColour);
        ctx.beginPath();
        for (const [a, b] of wf.edges) { ctx.moveTo(N[a][0], N[a][1]); ctx.lineTo(N[b][0], N[b][1]); }
        ctx.stroke();
      }
    }
    if (scene.displayNodes) {
      if (scene.useWireframeColour) scene.nodeColour = wf.colour;
      const dots = (pts, r, colour) => {
        if (r <= 0) return;
        ctx.fillStyle = colour; ctx.beginPath();
        for (const v of pts) { ctx.moveTo(Math.trunc(v[0]) + r, Math.trunc(v[1])); ctx.arc(Math.trunc(v[0]), Math.trunc(v[1]), r, 0, 2 * Math.PI); }
        ctx.fill();
      };
      if (scene.useRandomColour) for (const v of N) dots([v], nodeRadius, rgb(scene.getRandomColour()));
      else dots(N, nodeRadius, rgb(scene.nodeColour));
      if (scene.showHistory) {
        wf.history.forEach((old, count) => {
          const colour = scene.colourShift ? scene.shiftColour(scene.nodeColour, count) : scene.nodeColour;
          dots(old, Math.trunc(nodeRadius + count), rgb(colour)); // pygame ignores the alpha on the window surface
        });
      }
    }
  }
}
