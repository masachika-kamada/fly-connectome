"""Experiment 2 - how many odours can one mushroom body compartment hold?

The fly learns by depressing KC->MBON synapses: dopamine arriving in a
compartment weakens the synapses of whichever Kenyon cells were active. Learning
is subtractive, so after training the correct MBON receives the LEAST drive.

That gives a hard limit. Each odour lights a fraction f of the Kenyon cells, so
after k odours the share of a new odour's synapses already depressed is
1 - (1-f)^k. The analytic margin halves at k = 13.5 for f = 0.05; this measures
what actually happens on the real KC->MBON matrix.
"""
import json
import os

import numpy as np

from flymb import capacity, coding, data, odors, wiring
from flymb.paths import RESULTS, ensure

N_TEST = 240
K_VALUES = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 70, 100]
N_REPEAT = 8
SEED = 11
OUT = os.path.join(RESULTS, "exp2_capacity.json")


def main():
    w_pn_kc, w_kc_mbon, meta = data.load_mb()
    chan, n_chan = odors.channel_map(meta["pn"])
    n_kc, n_mbon = w_kc_mbon.shape

    usable, kc_per_mbon = capacity.usable_compartments(w_kc_mbon)
    print(f"KC x MBON        : {w_kc_mbon.shape}")
    print(f"KCs read per MBON: median {np.median(kc_per_mbon):.0f}  max {kc_per_mbon.max()}")
    print(f"usable (>={capacity.MIN_KC_INPUTS} KC): {len(usable)} of {n_mbon}")
    print(f"depression {coding.ETA:.2f}   sparsity {coding.SPARSITY}\n")

    rng0 = np.random.default_rng(SEED)
    variants = {
        "real": w_kc_mbon,
        "degree_swap": wiring.degree_preserving_swap(w_kc_mbon, rng0, passes=12),
        "uniform": wiring.uniform_random(w_kc_mbon, rng0),
    }

    results = {name: {k: [] for k in K_VALUES} for name in variants}
    for rep in range(N_REPEAT):
        rng = np.random.default_rng(SEED + 31 * rep)
        codes = capacity.coded_odors(max(K_VALUES) + N_TEST, w_pn_kc, chan, n_chan, rng)
        train, test = codes[:max(K_VALUES)], codes[max(K_VALUES):]
        mbon = int(rng.choice(usable))
        for name, W in variants.items():
            for k in K_VALUES:
                a = capacity.train_and_score(W, train, test, mbon, k)
                if a is not None:
                    results[name][k].append(a)

    f = coding.SPARSITY
    print(f"{'k':>4}  {'real':>14}  {'degree_swap':>14}  {'uniform':>14}   analytic (1-f)^k")
    print("-" * 78)
    summary = {}
    for k in K_VALUES:
        line = f"{k:>4}  "
        for name in variants:
            v = np.array(results[name][k])
            line += f"{v.mean():.3f}+/-{v.std():.3f}  "
            summary.setdefault(name, {})[k] = {"mean": float(v.mean()), "std": float(v.std())}
        print(line + f"    {(1-f)**k:.3f}")

    print("\ncapacity (largest k with AUC >= threshold):")
    caps = {}
    for name in variants:
        caps[name] = {}
        for thr in (0.95, 0.90, 0.80):
            ok = [k for k in K_VALUES if summary[name][k]["mean"] >= thr]
            caps[name][str(thr)] = max(ok) if ok else 0
            print(f"  {name:<12} AUC>={thr}: k = {caps[name][str(thr)]}")

    ensure(RESULTS)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({
            "dataset": meta["dataset"], "sparsity": f, "eta": coding.ETA,
            "n_kc": int(n_kc), "n_mbon": int(n_mbon),
            "usable_compartments": int(len(usable)),
            "repeats": N_REPEAT, "seed": SEED,
            "auc": summary, "capacity": caps,
            "analytic_margin": {str(k): (1 - f) ** k for k in K_VALUES},
        }, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
