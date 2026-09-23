"""Odour inputs, synthetic and measured."""
import numpy as np

# What each glomerulus is known to carry. Literature knowledge, not something
# derived from this dataset. FOOD and PHEROMONE only ever group results; the VP
# rule also decides which channels a synthetic odour may drive (the olfactory
# input model), but never touches the wiring.
FOOD = {"DP1m", "DP1l", "DC4", "DM1", "DM2", "DM4", "DM5", "VM2"}
PHEROMONE = {"DA1", "VA1v", "VA1d", "DL3"}


def is_thermo_hygro(glom):
    """VP glomeruli carry temperature and humidity, not odour (Marin et al. 2020)."""
    return glom.startswith("VP")


def glomerulus_role(glom):
    """'food', 'pheromone', 'thermo' or 'other'."""
    if is_thermo_hygro(glom):
        return "thermo"
    if glom in PHEROMONE:
        return "pheromone"
    if glom in FOOD:
        return "food"
    return "other"


def pn_glomerulus(p):
    """The glomerulus a uniglomerular PN belongs to, or None for a multiglomerular one."""
    t = p["type"] or "?"
    return None if t.startswith("M_") else t.split("_")[0]


def channel_map(meta_pn):
    """Map each ALPN to an input channel, returning (channel_per_pn, n_channels).

    Uniglomerular PNs share a channel with the other PNs of their glomerulus -
    that is what an odour actually addresses. Multiglomerular PNs (type starting
    'M_') each pool a different mixture, so each gets a channel of its own.

    Treating all 162 PNs as independent channels instead inflates the effective
    input dimension, which flatters random matrices. That was a real bug in the
    first version of experiment 1; both models are kept so the effect stays
    visible rather than hidden.
    """
    chan, index = [], {}
    for p in meta_pn:
        t = p["type"] or "?"
        key = f"multi:{p['bodyId']}" if t.startswith("M_") else f"glom:{t.split('_')[0]}"
        index.setdefault(key, len(index))
        chan.append(index[key])
    return np.array(chan), len(index)


def olfactory_channels(meta_pn, chan, n_chan):
    """Boolean mask over channels: True for a glomerulus that carries odour.

    Drops the 25 multiglomerular PNs and the thermo/hygrosensory VP glomeruli.
    In the glomerular model each multiglomerular PN is an independent channel
    as rich as a whole glomerulus, so those 25 PNs are over a quarter of the
    input dimensions - yet they carry about 1% of the synapses onto Kenyon
    cells. Scoring a hash on dimensions the circuit barely reads, and on
    temperature and humidity as if they were odours, charges the real wiring
    for ignoring things it has good reason to ignore.
    """
    keep = np.zeros(n_chan, dtype=bool)
    for p, c in zip(meta_pn, chan):
        g = pn_glomerulus(p)
        keep[c] = g is not None and not is_thermo_hygro(g)
    return keep


def synthetic_odors(n, chan, n_chan, rng, model="glomerular",
                    active=0.35, jitter=0.15, channels=None):
    """Generate n odour vectors in ALPN space.

    model="glomerular"     draw per glomerulus, broadcast to that glomerulus' PNs
    model="pn_independent" draw per PN (the naive model, kept as a control)

    `channels` (glomerular model only) is a boolean mask of the channels an
    odour may drive; the rest stay silent. The draws are made for every channel
    either way, so the same seed gives the same odours with and without a mask.
    """
    if model == "pn_independent":
        base = rng.lognormal(0.0, 1.0, size=(n, len(chan)))
        x = base * (rng.random((n, len(chan))) < active)
    elif model == "glomerular":
        base = rng.lognormal(0.0, 1.0, size=(n, n_chan))
        g = base * (rng.random((n, n_chan)) < active)
        if channels is not None:
            g[:, ~channels] = 0.0
        x = g[:, chan] * (1.0 + jitter * rng.normal(size=(n, len(chan))))
        x = np.maximum(x, 0)
    else:
        raise ValueError(f"unknown input model: {model}")
    # Divisive normalisation - the antennal lobe discards absolute concentration.
    x /= (x.sum(axis=1, keepdims=True) + 1e-9)
    return x.astype(np.float32)
