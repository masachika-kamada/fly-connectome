/* The errand: a fly is taught one smell, then walks a shop to fetch it.
 *
 * Shared by the /errand page and experiments/exp5_errand.js, so the fly the
 * page shows and the fly the experiment counts are the same fly.
 *
 * What is measured, and what is assumed:
 *   nose -> mushroom body   measured. DoOR responses in at the glomeruli, the
 *                           MaleCNS PN->KC wiring, 5% winner-take-all
 *   learning                measured wiring, standard rule. One pairing
 *                           weakens the lit KCs' synapses onto the MBONs of the
 *                           compartments the dopamine reaches
 *   air in the shop         assumed. Air conditioning blows from the back of
 *                           the shop to the door, so each product's smell
 *                           trails downwind as a plume that widens and fades;
 *                           where plumes overlap, the mixtures add linearly
 *   walking                 assumed, but only what walking flies are known to
 *                           do (Alvarez-Salvado et al. 2018): smell something
 *                           good and surge upwind, steering toward the antenna
 *                           that smells better; lose it and cast about.
 *                           At a shelf it judges the product itself - up
 *                           close, a product's own smell (and taste) swamps
 *                           whatever drifts over from its neighbours
 *
 * "Want" is the MBON population's verdict on the air at a point: reward
 * compartments' MBONs push away, punishment compartments' MBONs pull closer,
 * and training weakens whichever it reached. Naive, every smell is neutral:
 * the lateral horn, which carries innate likes and dislikes, is not modelled.
 */
(function (root) {
  'use strict';

  // Wind blows toward +y, from the back wall to the door. Two staggered rows
  // of shelves, so the back row's plumes pass between the front row's.
  var SHOP = { w: 1000, h: 640, entrance: { x: 500, y: 612 } };
  var SLOTS = [];
  for (var row = 0; row < 2; row++)
    for (var col = 0; col < 7; col++)
      SLOTS.push({ x: 100 + col * 133 + row * 67, y: 110 + row * 220 });

  var PARAMS = {
    plume: 16,         // width of a plume at its source
    spread: 0.16,      // how fast it widens per unit downwind
    cell: 12,          // resolution of the precomputed air, in shop units
    speed: 3,          // per step
    surgeSpeed: 4,     // walking upwind on a good smell
    antenna: 10,       // distance from body centre to each antenna
    splay: 0.6,        // angle of each antenna off the heading
    upwind: 0.25,      // pull of the heading toward upwind while surging
    steer: 0.6,        // turn per unit of normalised left-right difference
    wander: 0.12,      // heading noise with nothing worth following
    cast: 0.9,         // heading noise just after losing a good smell
    castSteps: 40,
    sense: 0.02,       // weakest smell a fly reacts to
    track: 0.9,        // how much a fly must want the air to follow it
    reach: 26,         // how close to a product counts as at it
    want: 1.4,         // how much a fly must want a product to take it
    maxSteps: 3000,
  };

  function mulberry32(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rand) {
    var u = 1 - rand(), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  // Rearranges a so that a[k] holds the k-th smallest value, and returns it.
  // Linear time on average; a full sort here made the shop take seconds.
  function select(a, k) {
    var l = 0, r = a.length - 1, t;
    while (r > l) {
      var m = (l + r) >> 1;
      if (a[m] < a[l]) { t = a[m]; a[m] = a[l]; a[l] = t; }
      if (a[r] < a[l]) { t = a[r]; a[r] = a[l]; a[l] = t; }
      if (a[r] < a[m]) { t = a[r]; a[r] = a[m]; a[m] = t; }
      var pivot = a[m], i = l, j = r;
      while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) { t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
      }
      if (k <= j) r = j; else if (k >= i) l = i; else break;
    }
    return a[k];
  }
  function shuffled(n, rand) {
    var a = []; for (var i = 0; i < n; i++) a.push(i);
    for (i = n - 1; i > 0; i--) { var j = Math.floor(rand() * (i + 1)), t = a[i]; a[i] = a[j]; a[j] = t; }
    return a;
  }

  // ---------------------------------------------------------------- brain
  function Brain(D, wiringKey) {
    var W = D.wirings[wiringKey];
    this.D = D;
    this.nKC = D.n_kc; this.K = D.kc_per_odor; this.eta = D.eta;
    this.ptr = Int32Array.from(W.ptr); this.pn = Int32Array.from(W.pn); this.syn = Float32Array.from(W.w);
    this.x = new Float64Array(D.n_pn);
    this.act = new Float64Array(this.nKC);
    this.sorted = new Float64Array(this.nKC);
    this.code = new Int32Array(this.K);
    this.w0 = D.compartments.map(function (c) { return Float64Array.from(c.w); });
    // Reward compartments' MBONs promote avoidance, so weakening them makes the
    // fly approach (+1); punishment compartments' MBONs promote approach (-1).
    this.sign = D.compartments.map(function (c) { return c.valence === 'reward' ? 1 : -1; });
    this.forget();
  }
  Brain.prototype.forget = function () {
    this.w = this.w0.map(function (a) { return Float64Array.from(a); });
    this.lessons = [];
  };
  // Mixture of glomerulus responses -> the K Kenyon cells that win. Returns
  // false when nothing is smelled at all.
  Brain.prototype.encode = function (glom) {
    var x = this.x, D = this.D, i, j, sum = 0;
    x.fill(0);
    for (i = 0; i < glom.length; i++) {
      var v = glom[i]; if (v <= 0) continue;
      var pns = D.pn_of_glom[i];
      for (j = 0; j < pns.length; j++) x[pns[j]] += v;
    }
    for (i = 0; i < x.length; i++) sum += x[i];
    if (sum <= 0) return false;
    // Divisive normalisation: the antennal lobe discards absolute concentration.
    for (i = 0; i < x.length; i++) x[i] /= sum;
    var act = this.act, ptr = this.ptr, pn = this.pn, syn = this.syn;
    for (var k = 0; k < this.nKC; k++) {
      var a = 0;
      for (j = ptr[k]; j < ptr[k + 1]; j++) a += x[pn[j]] * syn[j];
      act[k] = a;
    }
    this.sorted.set(act);
    var thr = select(this.sorted, this.nKC - this.K), n = 0;
    for (k = 0; k < this.nKC && n < this.K; k++) if (act[k] > thr) this.code[n++] = k;
    for (k = 0; k < this.nKC && n < this.K; k++) if (act[k] === thr) this.code[n++] = k;
    return true;
  };
  // How much the MBON population now pulls the fly toward this air. 0 when
  // untrained; one reward pairing takes the trained smell to about +2.5.
  Brain.prototype.want = function (glom) {
    if (!this.encode(glom)) return 0;
    var v = 0;
    for (var c = 0; c < this.w.length; c++) {
      var d0 = 0, d = 0, w0 = this.w0[c], w = this.w[c];
      for (var i = 0; i < this.K; i++) { var k = this.code[i]; d0 += w0[k]; d += w[k]; }
      if (d0 > 0) v += this.sign[c] * (1 - d / d0);
    }
    return v;
  };
  // One pairing of a product's smell with reward or punishment.
  Brain.prototype.teach = function (glom, kind) {
    if (!this.encode(glom)) return;
    for (var c = 0; c < this.w.length; c++) {
      if (this.D.compartments[c].valence !== kind) continue;
      for (var i = 0; i < this.K; i++) this.w[c][this.code[i]] *= (1 - this.eta);
    }
    this.lessons.push(kind);
  };

  // ---------------------------------------------------------------- shop air
  // One product's smell at (x, y): a plume trailing downwind (+y), widening
  // and thinning as it goes, with a little spill upwind of the shelf.
  function plume(src, x, y) {
    var dx = x - src.x, d = y - src.y, P = PARAMS;
    if (d <= 0) return Math.exp(-(dx * dx + 9 * d * d) / (2 * P.plume * P.plume));
    var sigma = P.plume + P.spread * d;
    return (P.plume / sigma) * Math.exp(-dx * dx / (2 * sigma * sigma));
  }
  // layout[s] = index of the product on shelf slot s.
  function airAt(D, layout, x, y, out) {
    var n = out.length, total = 0, i;
    for (i = 0; i < n; i++) out[i] = 0;
    for (var s = 0; s < layout.length; s++) {
      var c = plume(SLOTS[s], x, y);
      if (c < 1e-4) continue;
      var g = D.products[layout[s]].glom;
      for (i = 0; i < n; i++) out[i] += c * g[i];
      total += c;
    }
    return total;
  }

  // What the fly would make of the air everywhere in the shop, precomputed on
  // a grid: intensity, and how much it wants what it smells there.
  function Field(brain, layout) {
    var D = brain.D, cell = PARAMS.cell;
    this.nx = Math.floor(SHOP.w / cell) + 1; this.ny = Math.floor(SHOP.h / cell) + 1;
    this.intensity = new Float32Array(this.nx * this.ny);
    this.want = new Float32Array(this.nx * this.ny);
    var mix = new Float64Array(D.gloms.length);
    for (var j = 0; j < this.ny; j++)
      for (var i = 0; i < this.nx; i++) {
        var p = j * this.nx + i, I = airAt(D, layout, i * cell, j * cell, mix);
        this.intensity[p] = I;
        this.want[p] = brain.want(mix);
      }
    this.layout = layout.slice();
    // What the fly makes of each product up close, on its own.
    this.item = D.products.map(function (p) { return brain.want(p.glom); });
  }
  Field.prototype.sample = function (grid, x, y) {
    var cell = PARAMS.cell, fx = Math.max(0, Math.min(this.nx - 1.001, x / cell)),
        fy = Math.max(0, Math.min(this.ny - 1.001, y / cell));
    var i = Math.floor(fx), j = Math.floor(fy), u = fx - i, v = fy - j, n = this.nx;
    return (grid[j * n + i] * (1 - u) + grid[j * n + i + 1] * u) * (1 - v) +
           (grid[(j + 1) * n + i] * (1 - u) + grid[(j + 1) * n + i + 1] * u) * v;
  };
  // The steering signal: want, scaled by how strongly it is smelled.
  Field.prototype.pull = function (x, y) {
    return this.sample(this.want, x, y) * this.sample(this.intensity, x, y);
  };
  // Worth following: smelled at all, and wanted enough.
  Field.prototype.good = function (x, y) {
    return this.sample(this.intensity, x, y) > PARAMS.sense &&
           this.sample(this.want, x, y) > PARAMS.track;
  };

  // ---------------------------------------------------------------- fly
  function Fly(rand) {
    this.rand = rand;
    this.x = SHOP.entrance.x + (rand() - 0.5) * 40; this.y = SHOP.entrance.y;
    this.heading = -Math.PI / 2 + (rand() - 0.5) * 0.6;
    this.steps = 0; this.bought = null; this.done = false;
    this.lost = Infinity; this.trail = [];
    // For explaining an empty-handed return: did it ever follow a smell, and
    // which products did it stand at and turn down.
    this.surged = 0; this.passed = [];
  }
  Fly.prototype.step = function (field) {
    if (this.done) return;
    var P = PARAMS, h = this.heading, speed = P.speed;
    if (field.good(this.x, this.y)) {
      // Surge: head upwind (-y), and keep to the side that smells better.
      var aL = field.pull(this.x + P.antenna * Math.cos(h + P.splay), this.y + P.antenna * Math.sin(h + P.splay));
      var aR = field.pull(this.x + P.antenna * Math.cos(h - P.splay), this.y + P.antenna * Math.sin(h - P.splay));
      var s = Math.abs(aL) + Math.abs(aR);
      var up = Math.atan2(Math.sin(-Math.PI / 2 - h), Math.cos(-Math.PI / 2 - h));
      h += P.upwind * up + (s > 0 ? P.steer * (aL - aR) / s : 0) + P.wander * gauss(this.rand);
      speed = P.surgeSpeed; this.lost = 0; this.surged++;
    } else {
      // Just after losing a good smell, cast about (turn a lot); otherwise wander.
      this.lost++;
      h += (this.lost < P.castSteps ? P.cast : P.wander) * gauss(this.rand);
    }
    var nx = this.x + speed * Math.cos(h), ny = this.y + speed * Math.sin(h);
    if (nx < 8 || nx > SHOP.w - 8) { h = Math.PI - h; nx = Math.max(8, Math.min(SHOP.w - 8, nx)); }
    if (ny < 8 || ny > SHOP.h - 8) { h = -h; ny = Math.max(8, Math.min(SHOP.h - 8, ny)); }
    this.x = nx; this.y = ny; this.heading = h; this.steps++;
    if (this.steps % 3 === 0) this.trail.push(nx, ny);

    for (var k = 0; k < SLOTS.length; k++) {
      var dx = nx - SLOTS[k].x, dy = ny - SLOTS[k].y;
      if (dx * dx + dy * dy >= P.reach * P.reach) continue;
      var item = field.layout[k];
      if (field.item[item] > P.want) { this.bought = item; this.done = true; return; }
      if (this.passed.indexOf(item) < 0) this.passed.push(item);
    }
    if (this.steps >= P.maxSteps) this.done = true;
  };
  Fly.prototype.run = function (field) { while (!this.done) this.step(field); return this.bought; };

  var api = {
    SHOP: SHOP, SLOTS: SLOTS, PARAMS: PARAMS,
    mulberry32: mulberry32, shuffled: shuffled,
    Brain: Brain, Field: Field, Fly: Fly, airAt: airAt,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ErrandCore = api;
})(this);
