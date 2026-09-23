"""Stock the shop for the errand demo and experiment 5: data/errand.json.

A fly is shown one product's smell with a reward, then let loose in a shop to
fetch it. Everything between the nose and the mushroom body output is the same
as the rest of the project; what this adds is products, which smell of their
best-known aroma compounds.

Each product is approximated by one to three of the 67 DoOR chemicals, mixed in
equal parts. That is a simplification twice over - real aromas have dozens of
compounds, in ratios - so the products are labelled by their compounds in the
page, not presented as the real thing.

The page and experiment 5 both simulate the fly with web/errand-core.js, so
the full PN->KC matrix ships here (sparse, one copy per wiring) rather than
precomputed codes: the shop's air is a mixture that changes with position.
"""
import json
import os

import numpy as np

from flymb import capacity, coding, data as mbdata, door, wiring
from flymb.paths import DATA

OUT = os.path.join(DATA, "errand.json")
CONTROLS = os.path.join(DATA, "errand_controls.json")   # regenerable, not committed
WIRING_SEED = 1   # the same rewirings the trainer page offers
N_CONTROL = 19    # further rewirings of each kind: enough for a rank test at 5%

# name, icon, aroma compounds. The compound choices are textbook flavour
# chemistry; the pairings that share a compound (banana / sake, wine / sake)
# are deliberate, because that is where a fly - and a person - gets confused.
PRODUCTS = [
    ("バナナ", "🍌", ["isopentyl acetate"]),
    ("日本酒", "🍶", ["isopentyl acetate", "3-methyl-butanol"]),
    ("リンゴ", "🍎", ["hexyl acetate", "butyl acetate"]),
    ("パイナップル", "🍍", ["ethyl butyrate"]),
    ("ワイン", "🍷", ["ethyl acetate", "3-methyl-butanol", "phenethyl alcohol"]),
    ("お酢", "🫙", ["acetic acid"]),
    ("チーズ", "🧀", ["butyric acid", "isopentanoic acid"]),
    ("バター", "🧈", ["2,3-butanedione"]),
    ("キノコ", "🍄", ["1-octen-3-ol"]),
    ("バラ", "🌹", ["geraniol", "beta-citronellol", "phenethyl alcohol"]),
    ("杏仁豆腐", "🍮", ["benzaldehyde"]),
    ("ハチミツ", "🍯", ["phenylacetaldehyde"]),
    ("湿布", "🩹", ["methyl salicylate"]),
    ("生ゴミ", "🗑️", ["putrescine", "cadaverine"]),
]


def csr_by_kc(W):
    """PN->KC matrix as, for each KC, its input PNs and synapse counts."""
    ptr, pn, w = [0], [], []
    for k in range(W.shape[1]):
        rows = np.flatnonzero(W[:, k])
        pn.extend(int(r) for r in rows)
        w.extend(int(W[r, k]) for r in rows)
        ptr.append(len(pn))
    return {"ptr": ptr, "pn": pn, "w": w}


def main():
    W, Wm, meta = mbdata.load_mb()
    od = door.load(meta["pn"], log=lambda *_: None)
    index = {n: i for i, n in enumerate(od["names"])}

    products = []
    for name, icon, compounds in PRODUCTS:
        missing = [c for c in compounds if c not in index]
        if missing:
            raise SystemExit(f"{name}: not among the DoOR odorants: {missing}")
        g = np.mean([od["G"][index[c]] for c in compounds], axis=0)
        products.append({"name": name, "icon": icon, "compounds": compounds,
                         "glom": [round(float(v), 4) for v in g]})

    # glomerulus -> the PNs it drives
    pn_of_glom = [[int(j) for j in np.flatnonzero(row)] for row in od["P"]]

    rng = np.random.default_rng(WIRING_SEED)
    variants = {
        "real": W,
        "swap": wiring.degree_preserving_swap(W, rng),
        "uniform": wiring.uniform_random(W, rng),
    }

    comps = []
    for c in capacity.demo_compartments(Wm, meta["mbon"]):
        comps.append({k: c[k] for k in ("label", "type", "valence", "kc_inputs")}
                     | {"w": [int(v) for v in Wm[:, c["index"]]]})

    payload = {
        "source": {"connectome": meta["dataset"], "hemisphere": meta["hemisphere"],
                   "odors": "DoOR 2.0 (ropensci/DoOR.data)"},
        "n_pn": int(W.shape[0]),
        "n_kc": int(W.shape[1]),
        "kc_per_odor": max(1, int(round(coding.SPARSITY * W.shape[1]))),
        "eta": coding.ETA,
        "gloms": od["gloms"],
        "pn_of_glom": pn_of_glom,
        "wirings": {k: csr_by_kc(w) for k, w in variants.items()},
        "compartments": comps,
        "products": products,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))

    # One rewiring is one draw. Experiment 5 needs several of each kind to tell
    # a difference in the wiring from the luck of a single draw.
    rng = np.random.default_rng(WIRING_SEED + 1000)
    controls = {
        "swap": [csr_by_kc(wiring.degree_preserving_swap(W, rng)) for _ in range(N_CONTROL)],
        "uniform": [csr_by_kc(wiring.uniform_random(W, rng)) for _ in range(N_CONTROL)],
    }
    with open(CONTROLS, "w", encoding="utf-8") as fh:
        json.dump(controls, fh, separators=(",", ":"))
    print(f"wrote {CONTROLS}  ({N_CONTROL} rewirings of each kind)")
    print(f"{len(products)} products, {len(comps)} compartments, "
          f"{len(payload['wirings']['real']['pn'])} PN->KC edges per wiring")
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
