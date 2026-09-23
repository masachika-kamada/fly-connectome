"""Experiment 1 - is the real PN->KC wiring a better hash than random wiring?

Dasgupta, Stevens & Navlakha (Science 2017) showed the fly olfactory circuit
behaves like a locality-sensitive hash. "Random" was an assumption about the
anatomy; MaleCNS v1.0 lets us substitute the measured matrix and ask whether
evolution beat chance.

Metric: mean average precision of k-NN retrieval in hash space, scored against
the true cosine neighbours in odour space. Three input models are reported,
since how an odour addresses the glomeruli turned out to matter more than the
wiring:

  olfactory       per glomerulus, odour-carrying glomeruli only (the fair one)
  glomerular      per glomerulus, every channel including multiglomerular PNs
                  and the thermo/hygrosensory VP glomeruli
  pn_independent  per PN, the naive model

The gap between the first two is how much of "real loses to uniform" came
from scoring the circuit on channels it does not read.
"""
import collections
import json
import os

import numpy as np

from flymb import coding, data, odors, wiring
from flymb.paths import RESULTS, ensure

N_QUERY, N_POOL, K = 600, 2400, 50
N_REPEAT = 5
SEED = 7
OUT = os.path.join(RESULTS, "exp1_hash.json")


def main():
    w_pn_kc, _, meta = data.load_mb()
    n_pn, n_kc = w_pn_kc.shape
    chan, n_chan = odors.channel_map(meta["pn"])

    print(f"ALPN x KC      : {n_pn} x {n_kc} "
          f"(dropped {meta['kc_dropped']} KCs with no ALPN input)")
    print(f"input channels : {n_chan} (glomeruli + multiglomerular PNs)")
    kd = (w_pn_kc > 0).sum(axis=0)
    print(f"PN per KC      : mean {kd.mean():.2f}  "
          f"p5-p95 {np.percentile(kd,5):.0f}-{np.percentile(kd,95):.0f}")
    print(f"WTA keeps      : {int(round(coding.SPARSITY*n_kc))} of {n_kc}\n")

    olf = odors.olfactory_channels(meta["pn"], chan, n_chan)
    print(f"odour channels : {olf.sum()} of {n_chan} "
          f"(without multiglomerular PNs and thermo/hygro VP glomeruli)\n")

    rng = np.random.default_rng(SEED)
    variants = {
        "real": w_pn_kc,
        "binary_real": wiring.binary(w_pn_kc),
        "degree_swap": wiring.degree_preserving_swap(w_pn_kc, rng),
        "kc_degree_kept": wiring.kc_degree_kept(w_pn_kc, rng),
        "uniform": wiring.uniform_random(w_pn_kc, rng),
        "gaussian_lsh": wiring.dense_gaussian((n_pn, n_kc), rng),
    }
    # Drawn last so that adding it left every earlier control bit-identical.
    variants["pn_degree_kept"] = wiring.pn_degree_kept(w_pn_kc, rng)

    out = {}
    for model in ("olfactory", "glomerular", "pn_independent"):
        print(f"--- input model: {model} ---")
        scores = {name: [] for name in variants}
        for rep in range(N_REPEAT):
            r = np.random.default_rng(SEED + 100 * rep)
            if model == "olfactory":
                x = odors.synthetic_odors(N_POOL + N_QUERY, chan, n_chan, r,
                                          "glomerular", channels=olf)
            else:
                x = odors.synthetic_odors(N_POOL + N_QUERY, chan, n_chan, r, model)
            pool, query = x[:N_POOL], x[N_POOL:]
            truth = coding.true_neighbours(query, pool)
            for name, W in variants.items():
                scores[name].append(coding.mean_average_precision(
                    coding.hash_codes(query, W), coding.hash_codes(pool, W), truth, K))

        stats = {n: (float(np.mean(v)), float(np.std(v))) for n, v in scores.items()}
        base = stats["real"][0]
        for name, (m, s) in stats.items():
            rel = "" if name == "real" else f"   {100*(m-base)/base:+6.1f}% vs real"
            print(f"  {name:<14} mAP@{K} = {m:.4f} +/- {s:.4f}{rel}")
        print()
        out[model] = {n: {"mean": m, "std": s} for n, (m, s) in stats.items()}

    skew = outdegree_by_role(w_pn_kc, meta["pn"])
    print("KC edges per glomerulus, by what it is known to carry (median):")
    for role, v in skew["median"].items():
        print(f"  {role:<10} {v:>6.1f}   n={skew['n'][role]}")
    print(f"food+pheromone > other odours: one-sided permutation p = {skew['p_food_pheromone']:.3f}\n")

    ensure(RESULTS)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({
            "dataset": meta["dataset"], "hemisphere": meta["hemisphere"],
            "shape_full": [int(n_pn), int(n_kc) + meta["kc_dropped"]],
            "shape_used": [int(n_pn), int(n_kc)],
            "kc_dropped_no_alpn": meta["kc_dropped"],
            "input_channels": int(n_chan),
            "odour_channels": int(olf.sum()),
            "edges": int((w_pn_kc > 0).sum()),
            "pn_per_kc_mean": float(kd.mean()),
            "sparsity": coding.SPARSITY, "k": K, "repeats": N_REPEAT, "seed": SEED,
            "map": out,
            "glomerulus_outdegree": skew,
        }, fh, indent=1)
    print(f"wrote {OUT}")


def outdegree_by_role(w_pn_kc, meta_pn, n_perm=20000, seed=0):
    """KC edges per glomerulus, grouped by the literature role of the glomerulus.

    Tests the tempting story that the skew favours food and pheromone
    glomeruli, against the other odour glomeruli (thermo/hygro excluded: those
    are obviously low, and would make any comparison look significant).
    """
    per = collections.Counter()
    for p, d in zip(meta_pn, (w_pn_kc > 0).sum(axis=1)):
        g = odors.pn_glomerulus(p)
        if g:
            per[g] += int(d)
    role = {g: odors.glomerulus_role(g) for g in per}
    groups = {r: [per[g] for g in per if role[g] == r]
              for r in ("food", "pheromone", "other", "thermo")}
    fp = np.array(groups["food"] + groups["pheromone"], float)
    ot = np.array(groups["other"], float)
    both = np.concatenate([fp, ot])
    obs = fp.mean() - ot.mean()
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(both)
        hits += p[:len(fp)].mean() - p[len(fp):].mean() >= obs
    return {
        "per_glomerulus": dict(per.most_common()),
        "role": role,
        "n": {r: len(v) for r, v in groups.items()},
        "median": {r: float(np.median(v)) for r, v in groups.items()},
        "p_food_pheromone": (hits + 1) / (n_perm + 1),
    }


if __name__ == "__main__":
    main()
