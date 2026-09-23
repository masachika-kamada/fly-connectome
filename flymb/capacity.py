"""Training a single mushroom body compartment, and scoring what it retained.

Shared by experiments 2 and 3 so neither has to import the other.
"""
import re

import numpy as np

from . import coding, odors

MIN_KC_INPUTS = 100   # a compartment reading fewer Kenyon cells is not usable

# The compartments the demo offers, and what dopamine there means to the fly.
# Valence is literature knowledge (Aso et al., eLife 2014): PPL1 dopamine
# neurons signal punishment, PAM neurons reward. The learning rule is the same
# subtraction in both; what differs is the MBON's effect on behaviour.
# Depressing an approach-promoting MBON makes the fly avoid the odour;
# depressing an avoidance-promoting one makes it approach.
DEMO_COMPARTMENTS = [
    ("MBON11", "punish", "嫌悪記憶の研究で最もよく使われる区画"),
    ("MBON12", "punish", "短期の嫌悪記憶"),
    ("MBON19", "punish", "小さい区画。すぐに飽和する"),
    ("MBON02", "reward", "報酬の記憶"),
    ("MBON01", "reward", "γ5とβ'2aの2区画にまたがる報酬の区画"),
    ("MBON07", "reward", "長期の報酬記憶"),
]


def greek(instance):
    """'MBON11(y1pedc>a/B)_R' -> 'MBON11 γ1pedc>α/β'.

    neuPrint spells the lobes in ASCII: y, a, B. But a lowercase 'a' after a
    compartment number means anterior (β'2a), so 'a' only becomes α where a
    number, a prime or a slash follows it. Only the compartment code is
    touched, so a name like 'calyx' is left alone.
    """
    m = re.match(r"^(.*?)\((.*)\)(?:_[LR])?$", instance or "")
    if not m:
        return instance
    name, code = m.groups()
    if re.fullmatch(r"[yaB0-9'>/pedcms]+", code):
        code = re.sub(r"a(?=[0-9'/])", "α", code)
        code = re.sub(r"y(?=[0-9])", "γ", code).replace("B", "β")
    return f"{name} {code}"


def demo_compartments(w_kc_mbon, meta_mbon, side="R"):
    """Resolve DEMO_COMPARTMENTS to column indices: the largest instance per type."""
    kc_in = (w_kc_mbon > 0).sum(axis=0)
    out = []
    for mtype, valence, note in DEMO_COMPARTMENTS:
        cand = [j for j, m in enumerate(meta_mbon)
                if m["type"] == mtype and (m["instance"] or "").endswith("_" + side)]
        if not cand:
            raise ValueError(f"no {mtype} on side {side}")
        j = max(cand, key=lambda j: kc_in[j])
        out.append({"index": j, "type": mtype, "label": greek(meta_mbon[j]["instance"]),
                    "valence": valence, "note": note, "kc_inputs": int(kc_in[j])})
    return out


def coded_odors(n, w_pn_kc, chan, n_chan, rng, model="glomerular"):
    """Generate n odours and push them through the PN->KC expansion."""
    x = odors.synthetic_odors(n, chan, n_chan, rng, model)
    return coding.hash_codes(x, w_pn_kc)


def train_and_score(w_kc_mbon, codes_train, codes_test, mbon, k):
    """Write k memories into one compartment, return the separability AUC.

    Returns None if that compartment reads nothing.
    """
    w = w_kc_mbon[:, mbon].astype(np.float64)
    if w.sum() == 0:
        return None
    for i in range(k):
        w = coding.depress(w, codes_train[i])
    return coding.auc(coding.drive(codes_train[:k], w), coding.drive(codes_test, w))


def usable_compartments(w_kc_mbon, minimum=MIN_KC_INPUTS):
    kc_in = (w_kc_mbon > 0).sum(axis=0)
    return np.where(kc_in >= minimum)[0], kc_in
