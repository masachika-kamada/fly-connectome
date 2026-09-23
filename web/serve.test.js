const assert = require('node:assert/strict');
const http = require('node:http');
const path = require('node:path');
const test = require('node:test');
const handleRequest = require('./serve');

function request(server, pathname) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: '127.0.0.1', port: server.address().port, path: pathname, agent: false,
    };
    http.get(options, response => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', chunk => { body += chunk; });
      response.on('end', () => resolve({ status: response.statusCode, body }));
      response.on('error', reject);
    }).on('error', reject);
  });
}

test('serves project files and rejects escaped or malformed paths', async context => {
  const server = http.createServer(handleRequest);
  context.after(() => new Promise(resolve => server.close(resolve)));
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });

  // The flat names the published site uses, and the older pretty URL.
  for (const pathname of ['/explainer.html', '/explainer']) {
    const explainer = await request(server, pathname);
    assert.equal(explainer.status, 200, pathname);
    assert.match(explainer.body, /<canvas id="stage"/, pathname);
  }
  assert.equal((await request(server, '/figures.html')).status, 200);
  assert.equal((await request(server, '/figures/fig1.svg')).status, 200);

  const root = path.dirname(__dirname);
  const sibling = encodeURIComponent(path.basename(root) + '-other');
  const cases = [
    ['/web/../README.md', 200],
    ['/__missing_static_server_test__.txt', 404],
    ['/..not-a-parent.txt', 404],
    [`/../${sibling}/example.txt`, 403],
    [`/..%2f${sibling}/example.txt`, 403],
    ['/%2e%2e/README.md', 403],
    ['/%', 400],
    ['/%00', 400],
  ];
  if (process.platform === 'win32') {
    const otherDrive = path.parse(root).root[0].toUpperCase() === 'Z' ? 'Y' : 'Z';
    cases.push([`/..%5c${sibling}%5cexample.txt`, 403], [`/${otherDrive}:%5coutside.txt`, 403]);
  }
  cases.push(['/README.md', 200]);

  for (const [pathname, status] of cases) {
    const response = await request(server, pathname);
    assert.equal(response.status, status, pathname);
  }
});
// Each page carries its own copy of the site nav, so it still works opened as
// a plain file or published on its own. web/nav.html is the one to edit; this
// catches a page that was not updated to match.
test('every page carries the current site nav', () => {
  const fs = require('node:fs');
  const root = path.dirname(__dirname);
  const read = rel => fs.readFileSync(path.join(root, rel), 'utf8').replace(/\r\n/g, '\n');
  const nav = read('web/nav.html').trim();
  for (const page of ['web/trainer.template.html', 'web/artifact.html', 'web/figures.html']) {
    assert.ok(read(page).includes(nav), `${page} has a stale copy of web/nav.html`);
  }
});
