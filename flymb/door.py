"""Real odours: DoOR 2.0 consensus responses, mapped onto this mushroom body.

DoOR gives ~690 chemicals x ~78 receptors, normalised to [0,1].
door_mappings.csv says which glomerulus each receptor feeds, and MaleCNS gives
the ALPNs of each glomerulus. Chaining them:

  chemical -> receptors -> glomeruli -> ALPNs

Shared by the demo build and experiment 4, so both see exactly the same odours.
The raw tables are fetched by scripts/fetch_odors.py and not committed.
"""
import csv
import os

import numpy as np

from .odors import pn_glomerulus
from .paths import DATA

MIN_CHANNELS = 8        # an odorant needs at least this many measured channels
MIN_COVERAGE = 0.5      # ... and at least this share of the usable ones


def _read(name):
    with open(os.path.join(DATA, name), encoding="utf-8", errors="replace") as fh:
        return list(csv.reader(fh, delimiter=";", quotechar='"'))


def _float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def load(meta_pn, log=print):
    """Return the odours as a dict.

      names, classes  per odorant, sorted by name
      gloms           the glomeruli present in both datasets
      G               (n_odour, n_glom) mean response of each glomerulus
      P               (n_glom, n_pn) 1 where a PN belongs to that glomerulus
    """
    rm = _read("door_response_matrix.csv")
    receptors = [c.strip('"') for c in rm[0]]
    keys = [row[0] for row in rm[1:]]
    R = np.array([[_float(v) for v in row[1:]] for row in rm[1:]], dtype=float)
    log(f"DoOR matrix     : {R.shape[0]} odorants x {R.shape[1]} receptors")

    # The CSVs carry a row-name column the header does not, hence the +1s.
    od = _read("odor.csv")
    oh = [c.strip('"') for c in od[0]]
    ci, ni, ki = oh.index("Class"), oh.index("Name"), oh.index("InChIKey")
    info = {row[ki + 1]: {"name": row[ni + 1], "class": row[ci + 1]} for row in od[1:]}

    mp = _read("door_mappings.csv")
    mh = [c.strip('"') for c in mp[0]]
    r_i, g_i, o_i, c_i = (mh.index(k) for k in ("receptor", "glomerulus", "OSN", "code"))
    rec2glo = {}
    for row in mp[1:]:
        glo = row[g_i + 1]
        if glo in ("?", "", "NA"):
            continue
        for name in (row[r_i + 1], row[o_i + 1], row[c_i + 1]):
            if name and name not in ("?", "NA"):
                rec2glo[name] = glo
    log(f"receptor->glom  : {len(rec2glo)} entries")

    pn_glom = [pn_glomerulus(p) for p in meta_pn]
    cns_gloms = {g for g in pn_glom if g}
    log(f"MaleCNS gloms   : {len(cns_gloms)}")

    col_glom, cols = [], []
    for j, rec in enumerate(receptors):
        g = rec2glo.get(rec)
        if g and g in cns_gloms:
            col_glom.append(g)
            cols.append(j)
    gloms = sorted(set(col_glom))
    glom_ix = {g: i for i, g in enumerate(gloms)}
    log(f"matched gloms   : {len(gloms)}  ({len(cols)} DoOR columns feed them)")

    P = np.zeros((len(gloms), len(meta_pn)), dtype=np.float32)
    for i, g in enumerate(pn_glom):
        if g in glom_ix:
            P[glom_ix[g], i] = 1.0

    need = max(MIN_CHANNELS, int(MIN_COVERAGE * len(cols)))
    coverage = (~np.isnan(R[:, cols])).sum(axis=1)
    rows = np.where(coverage >= need)[0]
    log(f"odorants kept   : {len(rows)} (>= {need} measured channels)")

    # Several receptors can feed one glomerulus; average the ones measured.
    col_to_glom = np.array([glom_ix[g] for g in col_glom])
    odours = []
    for r in rows:
        v = R[r, cols]
        ok = ~np.isnan(v)
        s = np.bincount(col_to_glom[ok], weights=v[ok], minlength=len(gloms))
        n = np.bincount(col_to_glom[ok], minlength=len(gloms))
        vec = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        o = info.get(keys[r], {"name": keys[r], "class": "?"})
        if vec.sum() <= 0 or o["name"] in ("NA", "", "sfr"):
            continue
        cls = o["class"] if o["class"] not in ("NA", "") else "other"
        odours.append((o["name"], cls, vec))
    odours.sort(key=lambda t: t[0])

    return {
        "names": [t[0] for t in odours],
        "classes": [t[1] for t in odours],
        "gloms": gloms,
        "G": np.array([t[2] for t in odours]),
        "P": P,
    }


def pn_inputs(G, P):
    """Glomerulus responses -> ALPN activity, with divisive normalisation."""
    x = G @ P
    return (x / (x.sum(axis=1, keepdims=True) + 1e-9)).astype(np.float32)
