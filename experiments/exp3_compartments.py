"""Experiment 3 - capacity is a portfolio, not a single number.

Experiment 2 trains one compartment at a time and averages. But the mushroom
body's usable compartments differ enormously in how much of the Kenyon cell
population they read - from ~120 cells to over 1700. This measures each one
separately and checks whether capacity tracks that.

It also repeats experiment 2's real-vs-rewired comparison, paired by
compartment. Experiment 2 averages over randomly chosen compartments, so its
spread is mostly the size differences measured here; pairing removes that and
exposes a small, consistent difference that the unpaired spread hides.
"""
import json
import os

import numpy as np

from flymb import capacity, data, odors, wiring
from flymb.paths import RESULTS, ensure

K_VALUES = [5, 10, 20, 30, 50]
N_TEST = 240
N_REPEAT = 4
SEED = 5
N_SWAP = 3          # independent degree-preserving rewirings of KC->MBON
PAIRED_K = 20
OUT = os.path.join(RESULTS, "exp3_compartments.json")


def main():
    w_pn_kc, w_kc_mbon, meta = data.load_mb()
    chan, n_chan = odors.channel_map(meta["pn"])

    usable, kc_in = capacity.usable_compartments(w_kc_mbon)
    print(f"{len(usable)} compartments with >={capacity.MIN_KC_INPUTS} KC inputs "
          f"(range {kc_in[usable].min()}-{kc_in[usable].max()})\n")

    # Separate seeds, so the rewiring leaves the per-compartment numbers alone.
    swaps = [wiring.degree_preserving_swap(w_kc_mbon, np.random.default_rng(1000 + i), passes=12)
             for i in range(N_SWAP)]

    per = {int(m): {k: [] for k in K_VALUES} for m in usable}
    swapped = {int(m): [] for m in usable}
    for rep in range(N_REPEAT):
        rng = np.random.default_rng(SEED + 17 * rep)
        codes = capacity.coded_odors(max(K_VALUES) + N_TEST, w_pn_kc, chan, n_chan, rng)
        train, test = codes[:max(K_VALUES)], codes[max(K_VALUES):]
        for m in usable:
            for k in K_VALUES:
                a = capacity.train_and_score(w_kc_mbon, train, test, int(m), k)
                if a is not None:
                    per[int(m)][k].append(a)
            for w in swaps:
                swapped[int(m)].append(capacity.train_and_score(w, train, test, int(m), PAIRED_K))

    rows = []
    for m in usable:
        means = {k: float(np.mean(per[int(m)][k])) for k in K_VALUES}
        rows.append({
            "mbon": meta["mbon"][int(m)]["instance"],
            "type": meta["mbon"][int(m)]["type"],
            "kc_inputs": int(kc_in[m]),
            "auc_at_20": means[20],
            "auc_at_20_swapped": float(np.mean(swapped[int(m)])),
            "capacity_auc90": max([k for k in K_VALUES if means[k] >= 0.9] + [0]),
        })
    rows.sort(key=lambda r: -r["kc_inputs"])

    print(f"{'KC in':>6} {'cap':>5} {'AUC@20':>7}  compartment")
    print("-" * 56)
    for r in rows[:6]:
        print(f"{r['kc_inputs']:>6} {r['capacity_auc90']:>5} {r['auc_at_20']:>7.3f}  {r['mbon']}")
    print("   ...")
    for r in rows[-6:]:
        print(f"{r['kc_inputs']:>6} {r['capacity_auc90']:>5} {r['auc_at_20']:>7.3f}  {r['mbon']}")

    size = np.array([r["kc_inputs"] for r in rows], float)
    score = np.array([r["auc_at_20"] for r in rows], float)
    corr = float(np.corrcoef(np.log(size), score)[0, 1])
    big, small = score[size >= 500].mean(), score[size < 250].mean()
    print(f"\ncorrelation(log KC inputs, AUC@k=20) = {corr:+.3f}")
    print(f"AUC@k=20  >=500 KC inputs: {big:.3f}   <250: {small:.3f}")

    diff = np.array([r["auc_at_20_swapped"] - r["auc_at_20"] for r in rows])
    boot = np.random.default_rng(SEED).choice(diff, size=(10000, len(diff))).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nrewired minus real, AUC@k={PAIRED_K}, paired by compartment:")
    print(f"  mean {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"rewired higher in {np.sum(diff > 0)} of {len(diff)}")

    ensure(RESULTS)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"k_values": K_VALUES, "repeats": N_REPEAT, "seed": SEED,
                   "correlation_log_size_vs_auc20": corr,
                   "auc20_large": float(big), "auc20_small": float(small),
                   "paired_swap_minus_real": {
                       "k": PAIRED_K, "swaps": N_SWAP, "mean": float(diff.mean()),
                       "ci95": [float(lo), float(hi)],
                       "swap_higher": int(np.sum(diff > 0)), "n": int(len(diff))},
                   "compartments": rows}, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
