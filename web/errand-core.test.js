const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const test = require('node:test');
const E = require('./errand-core');

const DATA = path.join(path.dirname(__dirname), 'data', 'errand.json');
const D = fs.existsSync(DATA) ? JSON.parse(fs.readFileSync(DATA, 'utf8')) : null;
const skip = D ? false : 'data/errand.json not built (uv run scripts/build_errand.py)';

test('a code is exactly the winner-take-all share of Kenyon cells', { skip }, () => {
  const b = new E.Brain(D, 'real');
  assert.ok(b.encode(D.products[0].glom));
  assert.equal(new Set(b.code).size, D.kc_per_odor);
  assert.equal(b.encode(D.products[0].glom.map(() => 0)), false, 'no smell, no code');
});

test('one pairing moves the trained smell by eta in every compartment it reaches', { skip }, () => {
  const b = new E.Brain(D, 'real');
  const nReward = D.compartments.filter((c) => c.valence === 'reward').length;
  const nPunish = D.compartments.filter((c) => c.valence === 'punish').length;
  assert.equal(b.want(D.products[3].glom), 0, 'naive flies want nothing');
  b.teach(D.products[3].glom, 'reward');
  assert.ok(Math.abs(b.want(D.products[3].glom) - nReward * D.eta) < 1e-9);
  b.forget();
  b.teach(D.products[3].glom, 'punish');
  assert.ok(Math.abs(b.want(D.products[3].glom) + nPunish * D.eta) < 1e-9);
});

test('smells trail downwind, not up', () => {
  // One product with a single glomerulus, on the first shelf only.
  const src = E.SLOTS[0];
  const shop = { products: [{ glom: [1] }] };
  const at = (x, y) => E.airAt(shop, [0], x, y, new Float64Array(1));
  assert.ok(at(src.x, src.y + 150) > 20 * at(src.x, src.y - 150), 'downwind (+y) carries far more than upwind');
  assert.ok(at(src.x, src.y + 150) > at(src.x + 150, src.y + 150), 'the plume stays narrow');
});

test('an untrained fly fetches nothing, and a run is reproducible', { skip }, () => {
  const b = new E.Brain(D, 'real');
  const layout = E.shuffled(D.products.length, E.mulberry32(3));
  const naive = new E.Field(b, layout);
  for (let i = 0; i < 3; i++) assert.equal(new E.Fly(E.mulberry32(i)).run(naive), null);

  b.teach(D.products[3].glom, 'reward');
  const field = new E.Field(b, layout);
  const run = (seed) => { const f = new E.Fly(E.mulberry32(seed)); f.run(field); return [f.bought, f.steps]; };
  assert.deepEqual(run(11), run(11));
});
