"""Experiment 4 - the same questions, asked with real chemicals.

Experiments 1-3 use synthetic odours. This one pushes the 67 DoOR 2.0
chemicals the demo uses through every wiring control, and measures three
things, each of which the demo shows and docs/DEMO.md quotes:

  A. Does the KC code keep similar chemicals similar? Spearman correlation
     between receptor-level similarity and KC-code overlap, over every pair.
     With 67 odours a retrieval score like mAP@50 would be meaningless, so this
     replaces it. Random controls are drawn many times, so the real circuit can
     be placed within their distribution rather than against one draw.
  B. Punish one chemical: which others does the fly now avoid? Summarised over
     every chemical in turn, not one hand-picked example.
  C. How fast does a compartment fill up, under the demo's own readout?
     Averaged over random training orders, not one sequence of clicks.

Needs the DoOR tables (scripts/fetch_odors.py).
"""
import json
import os

import numpy as np

from flymb import capacity, coding, data, door, wiring
from flymb.paths import RESULTS, ensure

N_DRAWS = 20                         # random wirings per control
N_ORDERS = 500                       # random training orders for C
K_SATURATION = [5, 10, 15, 20, 30, 45]
TOP = 5                              # "most avoided" means the top 5
N_PERM = 2000
SEED = 23
EXAMPLE = "ethyl acetate"            # the worked example in docs/DEMO.md
OUT = os.path.join(RESULTS, "exp4_real_odors.json")


def cosine(x):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x @ x.T


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def similarity_preservation(codes, sim, iu):
    c = codes.astype(np.float32)
    return spearman(sim[iu], (c @ c.T)[iu])


def avoidance(codes, w0, i):
    """Punish odour i once; return every odour's fractional drop in drive."""
    w = coding.depress(w0, codes[i])
    base = coding.drive(codes, w0)
    return np.divide(base - coding.drive(codes, w), base, out=np.zeros(len(codes)), where=base > 0)


def generalisation(codes, w0, sim, classes):
    """Punish each odour alone, then summarise who else gets avoided.

    track_similarity  mean over odours of corr(avoidance, receptor similarity)
    class_enrichment  share of the top-5 most avoided that are the punished
                      odour's own chemical class, relative to chance
    """
    n = len(codes)
    profiles = np.array([avoidance(codes, w0, i) for i in range(n)])
    others = ~np.eye(n, dtype=bool)
    track = np.mean([np.corrcoef(profiles[i, others[i]], sim[i, others[i]])[0, 1]
                     for i in range(n)])
    return profiles, float(track), class_enrichment(profiles, classes)


def class_enrichment(profiles, classes):
    classes = np.asarray(classes)
    n = len(classes)
    hit, chance = [], []
    for i in range(n):
        same = classes == classes[i]
        if classes[i] == "other" or same.sum() < 3:
            continue
        order = [j for j in np.argsort(-profiles[i], kind="stable") if j != i][:TOP]
        hit.append(same[order].mean())
        chance.append((same.sum() - 1) / (n - 1))
    return float(np.mean(hit) / np.mean(chance))


def saturation(codes, w0, rng):
    """The demo's health bar, averaged over random training orders.

    A trained odour still counts as recognised while its drive is below the
    10th percentile of the untrained odours' drives - the rule the page uses.
    """
    n = len(codes)
    out = {k: [] for k in K_SATURATION}
    for _ in range(N_ORDERS):
        order = rng.permutation(n)[:max(K_SATURATION)]
        w = w0.copy()
        for step, i in enumerate(order, start=1):
            w = coding.depress(w, codes[i])
            if step in out:
                d = coding.drive(codes, w)
                trained = np.zeros(n, bool)
                trained[order[:step]] = True
                untr = np.sort(d[~trained])
                thr = untr[int(len(untr) * 0.1)]
                out[step].append(float(np.mean(d[trained] < thr)))
    return {str(k): {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in out.items()}


def worked_example(codes, w0, names, classes, profile):
    """The numbers behind the ethyl acetate walk-through in docs/DEMO.md."""
    i = names.index(EXAMPLE)
    readable = w0 > 0
    dep = codes[i] & readable
    rows = []
    for j in np.argsort(-profile, kind="stable"):
        lit = codes[j] & readable
        shared = lit & dep
        rows.append({"name": names[j], "cls": classes[j],
                     "drop": float(profile[j]),
                     "readable": int(lit.sum()), "shared": int(shared.sum()),
                     "synapse_share": float(w0[shared].sum() / w0[lit].sum())})
    return {"punished": EXAMPLE, "ranking": rows}


def main():
    w_pn_kc, w_kc_mbon, meta = data.load_mb()
    od = door.load(meta["pn"], log=lambda *_: None)
    names, classes = od["names"], od["classes"]
    x = door.pn_inputs(od["G"], od["P"])
    n = len(names)
    iu = np.triu_indices(n, 1)
    sim = cosine(x)
    print(f"{n} DoOR odorants over {len(od['gloms'])} glomeruli, {w_pn_kc.shape[1]} KCs\n")

    rng = np.random.default_rng(SEED)
    families = {
        "degree_swap": lambda: wiring.degree_preserving_swap(w_pn_kc, rng, passes=10),
        "pn_degree_kept": lambda: wiring.pn_degree_kept(w_pn_kc, rng),
        "kc_degree_kept": lambda: wiring.kc_degree_kept(w_pn_kc, rng),
        "uniform": lambda: wiring.uniform_random(w_pn_kc, rng),
    }
    draws = {name: [make() for _ in range(N_DRAWS)] for name, make in families.items()}

    # ---- A: similarity preservation
    codes_real = coding.hash_codes(x, w_pn_kc)
    reached = int((x @ w_pn_kc > 0).sum(axis=1).min())
    kept = int(codes_real.sum(axis=1)[0])
    print(f"fewest KCs with any input, over all odours: {reached} "
          f"(winner-take-all keeps {kept}"
          + (")\n" if reached >= kept else "; the rest are ties at zero)\n"))
    A = {"real": {"mean": similarity_preservation(codes_real, sim, iu), "std": 0.0},
         "binary_real": {"mean": similarity_preservation(
             coding.hash_codes(x, wiring.binary(w_pn_kc)), sim, iu), "std": 0.0}}
    for name, ws in draws.items():
        v = [similarity_preservation(coding.hash_codes(x, w), sim, iu) for w in ws]
        A[name] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                   "min": float(np.min(v)), "max": float(np.max(v)), "draws": v}
    swap = np.array(A["degree_swap"]["draws"])
    A["real_percentile_in_degree_swap"] = float(np.mean(swap <= A["real"]["mean"]))
    print("A. similarity preserved (Spearman, receptor similarity vs KC overlap)")
    for name, s in A.items():
        if isinstance(s, dict):
            print(f"  {name:<15} {s['mean']:.4f}" + (f" +/- {s['std']:.4f}" if s["std"] else ""))
    print(f"  real sits at the {100*A['real_percentile_in_degree_swap']:.0f}th percentile "
          f"of {N_DRAWS} degree-preserving rewirings\n")

    # ---- B: generalisation after punishing one odour
    comps = capacity.demo_compartments(w_kc_mbon, meta["mbon"])
    main_comp = comps[0]
    w0 = w_kc_mbon[:, main_comp["index"]].astype(np.float64)
    prof_real, track_real, enrich_real = generalisation(codes_real, w0, sim, classes)

    perm_rng = np.random.default_rng(SEED)
    null = [class_enrichment(prof_real, perm_rng.permutation(classes)) for _ in range(N_PERM)]
    p_enrich = (1 + sum(v >= enrich_real for v in null)) / (N_PERM + 1)

    swap_stats = []
    for w in draws["degree_swap"][:5]:
        c = coding.hash_codes(x, w)
        prof, track, enrich = generalisation(c, w0, sim, classes)
        same_cells = float(np.mean((c & codes_real).sum(axis=1) / codes_real.sum(axis=1)))
        agree = float(np.mean([np.corrcoef(prof[i], prof_real[i])[0, 1] for i in range(n)]))
        swap_stats.append((track, enrich, same_cells, agree))
    swap_stats = np.array(swap_stats).mean(axis=0)
    unif = [generalisation(coding.hash_codes(x, w), w0, sim, classes) for w in draws["uniform"][:5]]
    B = {
        "compartment": main_comp["label"],
        "real": {"track_similarity": track_real, "class_enrichment": enrich_real,
                 "class_enrichment_p": p_enrich},
        "degree_swap": {"track_similarity": float(swap_stats[0]),
                        "class_enrichment": float(swap_stats[1]),
                        "same_kc_as_real": float(swap_stats[2]),
                        "avoidance_corr_with_real": float(swap_stats[3])},
        "uniform": {"track_similarity": float(np.mean([u[1] for u in unif])),
                    "class_enrichment": float(np.mean([u[2] for u in unif]))},
        "chance_same_kc": float(codes_real.sum(axis=1)[0] / codes_real.shape[1]),
    }
    print(f"B. punish one odour in {main_comp['label']}, over all {n} in turn")
    print(f"  avoidance tracks receptor similarity: r = {track_real:.3f} "
          f"(rewired {swap_stats[0]:.3f}, uniform {B['uniform']['track_similarity']:.3f})")
    print(f"  own class in the top {TOP}: {enrich_real:.2f}x chance (p = {p_enrich:.4f})")
    print(f"  rewired wiring lights {100*swap_stats[2]:.1f}% of the same KCs "
          f"(chance {100*B['chance_same_kc']:.1f}%), yet its avoidance profile "
          f"correlates r = {swap_stats[3]:.3f} with the real one\n")

    # ---- C: saturation, per demo compartment
    C = {}
    print("C. share of trained odours still recognised (mean over "
          f"{N_ORDERS} random orders)")
    print(f"  {'':<22}" + "".join(f"{k:>7}" for k in K_SATURATION))
    for comp in comps:
        w0c = w_kc_mbon[:, comp["index"]].astype(np.float64)
        s = saturation(codes_real, w0c, np.random.default_rng(SEED))
        C[comp["label"]] = {"kc_inputs": comp["kc_inputs"], "valence": comp["valence"],
                            "recognised": s}
        print(f"  {comp['label']:<22}" + "".join(f"{s[str(k)]['mean']:>7.2f}" for k in K_SATURATION)
              + f"   ({comp['kc_inputs']} KC)")

    ensure(RESULTS)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({
            "n_odors": n, "gloms": len(od["gloms"]), "draws": N_DRAWS, "seed": SEED,
            "similarity_preservation": A,
            "generalisation": B,
            "saturation": {"k": K_SATURATION, "orders": N_ORDERS, "by_compartment": C},
            "example": worked_example(codes_real, w0, names, classes,
                                      prof_real[names.index(EXAMPLE)]),
        }, fh, ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
