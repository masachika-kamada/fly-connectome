"""Sparse coding, dopamine-gated learning, and the scoring metrics."""
import numpy as np

SPARSITY = 0.05   # fraction of Kenyon cells surviving winner-take-all (APL)
ETA = 0.85        # depression per pairing: the synapse drops to 15%


def hash_codes(x, W, sparsity=SPARSITY):
    """Project into Kenyon cell space, keep the top fraction, binarise.

    The winner-take-all stands in for APL-mediated global inhibition.
    """
    act = x @ W
    k = max(1, int(round(sparsity * W.shape[1])))
    idx = np.argpartition(-act, k - 1, axis=1)[:, :k]
    codes = np.zeros(act.shape, dtype=bool)
    np.put_along_axis(codes, idx, True, axis=1)
    return codes


def depress(weights, code, eta=ETA):
    """One pairing: dopamine weakens the synapses of the cells that were active.

    Returns a new vector; the caller keeps the running state. Learning is
    subtractive, which is why the trained odour ends up driving its compartment
    LESS than an untrained one.
    """
    out = weights.copy()
    out[code] *= (1.0 - eta)
    return out


def drive(codes, weights):
    """Total input each coded odour delivers to one compartment."""
    return codes.astype(np.float64) @ weights


def auc(trained, novel):
    """P(a trained odour scores below a novel one), ties counted as half.

    Learning depresses, so lower is "recognised" - hence trained < novel.
    """
    allv = np.concatenate([trained, novel])
    order = allv.argsort(kind="stable")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_t = len(trained)
    r_t = ranks[:n_t].sum()
    return float((n_t * (n_t + 1) / 2 + n_t * len(novel) - r_t) / (n_t * len(novel)))


def mean_average_precision(codes_q, codes_p, truth_rank, k=50):
    """mAP@k of hash-space retrieval, scored against the true neighbour ranking.

    Hash similarity is the overlap between two sparse codes.
    """
    sim = codes_q.astype(np.float32) @ codes_p.astype(np.float32).T
    aps = np.empty(sim.shape[0])
    for i in range(sim.shape[0]):
        relevant = set(truth_rank[i, :k].tolist())
        order = np.argsort(-sim[i], kind="stable")[:k]
        hits, score = 0, 0.0
        for rank, cand in enumerate(order, start=1):
            if cand in relevant:
                hits += 1
                score += hits / rank
        aps[i] = score / k
    return float(aps.mean())


def true_neighbours(query, pool):
    """Cosine-similarity ranking of the pool for each query - the ground truth."""
    p = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-9)
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
    return np.argsort(-(q @ p.T), axis=1)
