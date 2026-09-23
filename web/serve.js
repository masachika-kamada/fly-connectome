const http = require('http'), fs = require('fs'), path = require('path');
const ROOT = path.dirname(__dirname);
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
  '.svg': 'image/svg+xml; charset=utf-8', '.json': 'application/json; charset=utf-8',
  '.md': 'text/plain; charset=utf-8', '.png': 'image/png'
};
const PORT = 5173;

function handleRequest(req, res) {
  let rel;
  try {
    rel = decodeURIComponent(req.url.split('?')[0]);
  } catch {
    res.writeHead(400); res.end('bad request'); return;
  }
  if (rel.includes('\0')) { res.writeHead(400); res.end('bad request'); return; }
  // The same flat names the published site uses (scripts/build_pages.py
  // assembles site/ with them), so relative links behave identically here and
  // under GitHub Pages. The older pretty URLs keep working.
  const ROUTES = {
    '/': 'dist/errand.html',             // the errand leads: it is the strongest piece
    '/index.html': 'dist/errand.html',
    '/brain.html': 'dist/trainer.html',  // look inside the mushroom body
    '/explainer.html': 'web/artifact.html',
    '/figures.html': 'web/figures.html',
    '/errand': 'dist/errand.html',
    '/trainer': 'dist/trainer.html',
    '/brain': 'dist/trainer.html',
    '/explainer': 'web/artifact.html',
    '/figures': 'web/figures.html',
  };
  rel = ROUTES[rel.replace(/\/$/, '') || '/'] || rel.replace(/^\//, '');
  const file = path.resolve(ROOT, rel);
  const relative = path.relative(ROOT, file);
  if (relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) {
    res.writeHead(403); res.end('nope'); return;
  }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('not found'); return; }
    // No caching: this only ever serves a working copy, and a stale page here
    // costs more debugging time than the bytes it saves.
    res.writeHead(200, {
      'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
      'Cache-Control': 'no-store, max-age=0',
    });
    res.end(data);
  });
}

if (require.main === module) {
  http.createServer(handleRequest).listen(PORT, () => {
    console.log(`preview on http://localhost:${PORT}`);
    console.log('  /                 the errand: a fly fetches a product');
    console.log('  /brain.html       inside the mushroom body');
    console.log('  /explainer.html   explainer animation');
    console.log('  /figures.html     result figures');
  });
}

module.exports = handleRequest;
