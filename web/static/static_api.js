// Static-site data layer: answers api() calls from the JSON files exported by scripts/build_static.py.
// /shape/<slug> is exported only for the shapes the browser cannot build (SciPy / scikit-image); the others are built
// by shapes_engine.js. /check/marching_cubes is left out of the export for size and needs the local app.
const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => {
    if (!r.ok) throw new Error(r.status === 404 ? `${rel} is only available in the local app (Python server)` : `${r.status} ${rel}`);
    return r.json();
  }));
  return files.get(rel);
}

export async function staticApi(path) {
  return file(path.split('?')[0].replace(/^\/+/, ''));
}
