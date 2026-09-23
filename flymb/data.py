"""Load the cached mushroom body subgraph."""
import json

import numpy as np

from .paths import MB_META, MB_NPZ


def load_mb(drop_non_olfactory=True):
    """Return (W_pn_kc, W_kc_mbon, meta).

    128 of the 2045 Kenyon cells receive no ALPN input at all - they are KCg-d
    (visual) and KCab-p. In an olfactory experiment they only waste
    winner-take-all budget on cells that can never respond, so they are dropped
    by default. Both matrices are filtered consistently.
    """
    d = np.load(MB_NPZ)
    w_pn_kc, w_kc_mbon = d["W_pn_kc"], d["W_kc_mbon"]
    with open(MB_META, encoding="utf-8") as fh:
        meta = json.load(fh)
    if drop_non_olfactory:
        keep = (w_pn_kc > 0).sum(axis=0) > 0
        w_pn_kc, w_kc_mbon = w_pn_kc[:, keep], w_kc_mbon[keep, :]
        meta["kc_dropped"] = int((~keep).sum())
    return w_pn_kc, w_kc_mbon, meta
