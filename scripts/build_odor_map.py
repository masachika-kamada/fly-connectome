"""Join DoOR odorant responses to the MaleCNS mushroom body, for the demo.

  chemical -> receptors -> glomeruli -> ALPNs -> Kenyon cells

The chain up to the ALPNs lives in flymb.door, shared with experiment 4, so
the demo and the numbers quoted about it come from the same odours.

Besides the real wiring, the page gets the Kenyon cell codes under two of
experiment 1's controls, so a visitor can swap the wiring out from under a
trained fly and see what changes.

Writes data/odor_kc.json for the browser demo.
"""
import json
import os

import numpy as np

from flymb import capacity, coding, data as mbdata, door, wiring
from flymb.paths import ODOR_KC as OUT

WIRING_SEED = 1


def main():
    W, Wm, meta = mbdata.load_mb()
    od = door.load(meta["pn"])
    print("  ", ", ".join(od["gloms"]))
    x = door.pn_inputs(od["G"], od["P"])

    rng = np.random.default_rng(WIRING_SEED)
    variants = {
        "real": W,
        "swap": wiring.degree_preserving_swap(W, rng),
        "uniform": wiring.uniform_random(W, rng),
    }
    codes = {k: [sorted(int(i) for i in np.flatnonzero(row)) for row in coding.hash_codes(x, w)]
             for k, w in variants.items()}

    classes = {}
    for c in od["classes"]:
        classes[c] = classes.get(c, 0) + 1
    print(f"\nfinal odorants  : {len(od['names'])}")
    print("by chemical class:", dict(sorted(classes.items(), key=lambda kv: -kv[1])[:10]))

    comps = []
    for c in capacity.demo_compartments(Wm, meta["mbon"]):
        comps.append({k: c[k] for k in ("label", "type", "valence", "note", "kc_inputs")}
                     | {"w": [int(v) for v in Wm[:, c["index"]]]})
    print("\ncompartments offered:")
    for c in comps:
        print(f"  {c['type']:<8} {c['valence']:<7} {c['kc_inputs']:>5} KC inputs")

    payload = {
        "source": {
            "connectome": meta["dataset"], "hemisphere": meta["hemisphere"],
            "odors": "DoOR 2.0 (ropensci/DoOR.data)",
        },
        "n_pn": int(W.shape[0]),
        "n_kc": int(W.shape[1]),
        "sparsity": coding.SPARSITY,
        "eta": coding.ETA,
        "kc_per_odor": len(codes["real"][0]),
        "gloms": od["gloms"],
        "odors": [{"name": n, "cls": c, "glom": [round(float(v), 4) for v in g]}
                  for n, c, g in zip(od["names"], od["classes"], od["G"])],
        "codes": codes,
        "compartments": comps,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
