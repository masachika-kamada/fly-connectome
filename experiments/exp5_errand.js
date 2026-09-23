/* Experiment 5 - the errand: taught one smell, can a fly fetch it from a shop?
 *
 * A fly smells a product once with a reward, then walks a shop of 14 products
 * whose smells trail downwind and mix. It takes the first product it stands
 * at and wants enough, judging the product itself rather than the mixed air. The fly is web/errand-core.js, the same code the /errand page runs.
 *
 *   A. What does it bring back? Every product as the errand, shelves shuffled.
 *   B. Does the real wiring matter? The same errands, on rewired brains.
 *   C. Can it be taught out of a mistake? Reward the errand, punish the thing
 *      it most often fetches instead - differential conditioning, which in
 *      real flies sharpens discrimination (Barth et al. 2014).
 *
 * Needs data/errand.json and data/errand_controls.json (scripts/build_errand.py).
 * Takes a few minutes.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const E = require('../web/errand-core.js');

const ROOT = path.dirname(__dirname);
const D = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'errand.json'), 'utf8'));
const CONTROLS = path.join(ROOT, 'data', 'errand_controls.json');
const OUT = path.join(ROOT, 'results', 'exp5_errand.json');

const LAYOUTS = 10;          // shelf arrangements per errand (A, C)
const LAYOUTS_B = 3;         // per errand per rewired brain (B)
const FLIES = 20;            // flies per arrangement
const SEED = 29;

const names = D.products.map((p) => p.name);
const N = names.length;

function layoutFor(i) { return E.shuffled(N, E.mulberry32(SEED * 1000 + i)); }

// One errand: returns counts of what came back, index N meaning nothing.
function errand(brain, target, layouts, avoid) {
  const counts = new Array(N + 1).fill(0);
  for (let L = 0; L < layouts; L++) {
    brain.forget();
    brain.teach(D.products[target].glom, 'reward');
    if (avoid != null) brain.teach(D.products[avoid].glom, 'punish');
    const field = new E.Field(brain, layoutFor(L));
    for (let f = 0; f < FLIES; f++) {
      const got = new E.Fly(E.mulberry32(SEED + 7919 * L + 104729 * target + f)).run(field);
      counts[got == null ? N : got]++;
    }
  }
  return counts;
}

function wilson(k, n) {
  const z = 1.96, p = k / n, d = 1 + z * z / n;
  const c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [c - h, c + h];
}
function pearson(a, b) {
  const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return sab / Math.sqrt(saa * sbb);
}
const pct = (x) => (100 * x).toFixed(0) + '%';

function main() {
  if (!fs.existsSync(CONTROLS)) {
    console.error(`missing ${CONTROLS}: run  uv run scripts/build_errand.py`);
    process.exit(1);
  }
  const controls = JSON.parse(fs.readFileSync(CONTROLS, 'utf8'));
  const P = E.PARAMS;
  console.log(`${N} products, ${LAYOUTS} shelf arrangements x ${FLIES} flies per errand\n`);

  // ---- A: what comes back
  const real = new E.Brain(D, 'real');
  const confusion = [];
  for (let t = 0; t < N; t++) confusion.push(errand(real, t, LAYOUTS));
  const per = LAYOUTS * FLIES;
  const success = confusion.map((row, t) => row[t] / per);
  const none = confusion.map((row) => row[N] / per);
  const hits = confusion.reduce((s, row, t) => s + row[t], 0), total = per * N;
  console.log('A. what the fly brings back');
  confusion.forEach((row, t) => {
    const wrong = row.map((c, j) => [c, j]).filter(([c, j]) => j !== t && j < N && c > 0)
      .sort((a, b) => b[0] - a[0]).slice(0, 2).map(([c, j]) => `${names[j]} ${pct(c / per)}`);
    console.log(`  ${names[t].padEnd(7, '　')} right ${pct(success[t]).padStart(4)}  nothing ${pct(none[t]).padStart(4)}  ${wrong.join(', ')}`);
  });
  const ci = wilson(hits, total);
  console.log(`  overall ${pct(hits / total)} (95% CI ${pct(ci[0])}-${pct(ci[1])}); `
    + `picking a shelf at random would be ${pct(1 / N)}\n`);

  // Errors should follow what the fly learned to want. want[t][j]: after one
  // reward on t, how much the fly wants product j smelled on its own.
  const want = [];
  for (let t = 0; t < N; t++) {
    real.forget(); real.teach(D.products[t].glom, 'reward');
    want.push(D.products.map((p) => real.want(p.glom)));
  }
  const xs = [], ys = [];
  for (let t = 0; t < N; t++) for (let j = 0; j < N; j++) if (j !== t) { xs.push(want[t][j]); ys.push(confusion[t][j] / per); }
  const r = pearson(xs, ys);
  console.log(`   wrong purchases track learned want: r = ${r.toFixed(2)} over ${xs.length} product pairs\n`);

  // ---- B: rewired brains, same errands, same shelves, same flies
  function rate(brain) {
    let k = 0;
    for (let t = 0; t < N; t++) k += errand(brain, t, LAYOUTS_B)[t];
    return k / (N * LAYOUTS_B * FLIES);
  }
  const B = { real: rate(real) };
  for (const kind of ['swap', 'uniform']) {
    const draws = controls[kind].map((w) => {
      const d = Object.assign({}, D, { wirings: { x: w } });
      return rate(new E.Brain(d, 'x'));
    });
    const m = draws.reduce((s, v) => s + v, 0) / draws.length;
    const sd = Math.sqrt(draws.reduce((s, v) => s + (v - m) ** 2, 0) / draws.length);
    // If the real wiring were just another draw, it would rank anywhere among
    // the n + 1 brains with equal chance: this is that rank test.
    const atLeast = draws.filter((v) => v >= B.real).length;
    B[kind] = { draws, mean: m, sd, below_real: draws.length - atLeast,
                p_rank: (1 + atLeast) / (draws.length + 1) };
  }
  console.log(`B. errands fetched correctly (${LAYOUTS_B} arrangements x ${FLIES} flies x ${N} errands per brain)`);
  console.log(`  real      ${pct(B.real)}`);
  for (const kind of ['swap', 'uniform'])
    console.log(`  ${kind.padEnd(9)} ${pct(B[kind].mean)} +/- ${pct(B[kind].sd)}  `
      + `(${B[kind].below_real} of ${B[kind].draws.length} below real, rank p = ${B[kind].p_rank.toFixed(3)})`);
  console.log();

  // ---- C: punish the usual mistake
  const C = [];
  for (let t = 0; t < N; t++) {
    let worst = -1;
    for (let j = 0; j < N; j++) if (j !== t && (worst < 0 || confusion[t][j] > confusion[t][worst])) worst = j;
    if (confusion[t][worst] / per < 0.05) continue;
    const after = errand(real, t, LAYOUTS, worst);
    // The gap between the two is what discrimination is; the level of the
    // errand itself is what decides whether the fly buys anything at all.
    real.forget(); real.teach(D.products[t].glom, 'reward'); real.teach(D.products[worst].glom, 'punish');
    C.push({ target: t, avoid: worst, before: success[t], after: after[t] / per,
             mistake_before: confusion[t][worst] / per, mistake_after: after[worst] / per,
             nothing_after: after[N] / per,
             want_before: [want[t][t], want[t][worst]],
             want_after: [real.want(D.products[t].glom), real.want(D.products[worst].glom)] });
  }
  console.log('C. reward the errand, punish the usual mistake');
  for (const c of C)
    console.log(`  ${names[c.target]} (not ${names[c.avoid]}): right ${pct(c.before)} -> ${pct(c.after)}, `
      + `wrong one ${pct(c.mistake_before)} -> ${pct(c.mistake_after)}, nothing ${pct(c.nothing_after)}; `
      + `want ${c.want_before.map((v) => v.toFixed(2)).join('/')} -> ${c.want_after.map((v) => v.toFixed(2)).join('/')}`);

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify({
    params: P, layouts: LAYOUTS, layouts_b: LAYOUTS_B, flies: FLIES, seed: SEED,
    products: D.products.map((p) => ({ name: p.name, icon: p.icon, compounds: p.compounds })),
    confusion, success, none, overall: hits / total, overall_ci95: ci, chance: 1 / N,
    want, want_vs_mistakes_r: r, wiring: B, differential: C,
  }, null, 1));
  console.log(`\nwrote ${OUT}`);
}

main();
