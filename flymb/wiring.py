"""Randomisation controls.

Each one destroys a different property of the measured matrix, so comparing them
says which property the circuit's performance actually rests on.
"""
import numpy as np


def degree_preserving_swap(W, rng, passes=20):
    """Rewire every edge while keeping both degree sequences exactly.

    Double-edge swap. If performance survives this, the individual
    glomerulus-to-KC correspondences carry no information - only the degrees do.
    """
    rows, cols = np.nonzero(W)
    vals = W[rows, cols].copy()
    rows, cols = rows.copy(), cols.copy()
    present = {(r, c) for r, c in zip(rows, cols)}
    m = len(rows)
    for _ in range(passes * m):
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        r1, c1, r2, c2 = rows[i], cols[i], rows[j], cols[j]
        if r1 == r2 or c1 == c2 or (r1, c2) in present or (r2, c1) in present:
            continue
        present.discard((r1, c1))
        present.discard((r2, c2))
        present.add((r1, c2))
        present.add((r2, c1))
        cols[i], cols[j] = c2, c1
    out = np.zeros_like(W)
    out[rows, cols] = vals
    return out


def uniform_random(W, rng):
    """Same edge count and same weight pool, placed uniformly at random.

    Destroys both degree sequences.
    """
    n_edges = int((W > 0).sum())
    out = np.zeros_like(W)
    out.flat[rng.choice(W.size, size=n_edges, replace=False)] = rng.permutation(W[W > 0])
    return out


def kc_degree_kept(W, rng):
    """Keep each column's in-degree, but draw its partners uniformly.

    The control that separates the two degree effects. KC in-degree (how many
    glomeruli one Kenyon cell samples) stays as measured; PN out-degree (how many
    Kenyon cells one glomerulus reaches) is flattened. If this recovers the
    `uniform` score, the gap is out-degree skew and nothing else.
    """
    n_pre, n_post = W.shape
    out = np.zeros_like(W)
    vals = rng.permutation(W[W > 0])
    at = 0
    for j in range(n_post):
        k = int((W[:, j] > 0).sum())
        if k == 0:
            continue
        out[rng.choice(n_pre, size=k, replace=False), j] = vals[at:at + k]
        at += k
    return out


def pn_degree_kept(W, rng):
    """Keep each row's out-degree and its own weights, draw the partners uniformly.

    The mirror image of kc_degree_kept. If this matches `real`, the PN side
    (how many Kenyon cells each input reaches, and how strongly) is sufficient
    to explain the real circuit's score, and the KC side adds nothing.
    """
    n_pre, n_post = W.shape
    out = np.zeros_like(W)
    for i in range(n_pre):
        vals = W[i][W[i] > 0]
        if len(vals):
            out[i, rng.choice(n_post, size=len(vals), replace=False)] = rng.permutation(vals)
    return out


def binary(W):
    """Real edge placement, every weight set to 1.

    Tests whether synapse count is a good stand-in for synaptic weight.
    """
    return (W > 0).astype(np.float32)


def dense_gaussian(shape, rng):
    """Classic LSH baseline: a dense random projection."""
    return np.abs(rng.normal(size=shape)).astype(np.float32)
