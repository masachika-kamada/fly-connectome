"""Pull the right-hemisphere mushroom body subgraph out of MaleCNS v1.0.

Source: neuPrint public Cypher endpoint (no auth needed for /api/custom/custom).
Dataset: male-cns:v1.0  (Janelia / Google Research, Sep 2026)

Writes data/mb_right.npz and data/mb_meta.json. Safe to re-run; it skips the
network entirely if the cache is already there (pass --force to refetch).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import requests

from flymb.paths import DATA, MB_META, MB_NPZ, ensure

API = "https://neuprint.janelia.org/api/custom/custom"
DATASET = "male-cns:v1.0"
NPZ, META = MB_NPZ, MB_META

SIDE = "R"  # one hemisphere, so the KC population is a single mushroom body


def cypher(query, retries=3):
    """POST one Cypher query, with a polite retry."""
    for attempt in range(retries):
        try:
            r = requests.post(
                API, json={"dataset": DATASET, "cypher": query}, timeout=300
            )
            r.raise_for_status()
            return r.json()["data"]
        except Exception as exc:  # noqa: BLE001 - surface anything and retry
            if attempt == retries - 1:
                raise
            wait = 3 * (attempt + 1)
            print(f"  retry in {wait}s ({exc})", file=sys.stderr)
            time.sleep(wait)
    return []


def fetch():
    print(f"dataset: {DATASET}   hemisphere: {SIDE}")

    print("[1/3] node metadata ...")
    nodes = cypher(
        "MATCH (n:Neuron) WHERE n.class IN ['ALPN','Kenyon_Cell','MBON','DAN'] "
        "RETURN n.bodyId, n.type, n.class, n.somaSide, n.instance"
    )
    print(f"      {len(nodes)} neurons")

    # KCs define the hemisphere. PN and MBON partners are taken as they come,
    # because some of them are genuinely bilateral.
    kc_ids = [n[0] for n in nodes if n[2] == "Kenyon_Cell" and n[3] == SIDE]
    print(f"      {len(kc_ids)} Kenyon cells on side {SIDE}")

    print("[2/3] ALPN -> KC edges ...")
    pn_kc = cypher(
        "MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron) "
        f"WHERE a.class='ALPN' AND b.class='Kenyon_Cell' AND b.somaSide='{SIDE}' "
        "RETURN a.bodyId, b.bodyId, w.weight"
    )
    print(f"      {len(pn_kc)} edges")

    print("[3/3] KC -> MBON edges ...")
    kc_mbon = cypher(
        "MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron) "
        f"WHERE a.class='Kenyon_Cell' AND a.somaSide='{SIDE}' AND b.class='MBON' "
        "RETURN a.bodyId, b.bodyId, w.weight"
    )
    print(f"      {len(kc_mbon)} edges")

    return nodes, pn_kc, kc_mbon


def build(nodes, pn_kc, kc_mbon):
    """Turn the edge lists into dense weight matrices with stable index order."""
    by_id = {n[0]: {"type": n[1], "class": n[2], "side": n[3], "instance": n[4]}
             for n in nodes}

    kc_ids = sorted(i for i, n in by_id.items()
                    if n["class"] == "Kenyon_Cell" and n["side"] == SIDE)
    pn_ids = sorted({e[0] for e in pn_kc})
    mbon_ids = sorted({e[1] for e in kc_mbon})

    kc_ix = {b: i for i, b in enumerate(kc_ids)}
    pn_ix = {b: i for i, b in enumerate(pn_ids)}
    mbon_ix = {b: i for i, b in enumerate(mbon_ids)}

    W_pn_kc = np.zeros((len(pn_ids), len(kc_ids)), dtype=np.float32)
    for pre, post, w in pn_kc:
        W_pn_kc[pn_ix[pre], kc_ix[post]] = w

    W_kc_mbon = np.zeros((len(kc_ids), len(mbon_ids)), dtype=np.float32)
    for pre, post, w in kc_mbon:
        W_kc_mbon[kc_ix[pre], mbon_ix[post]] = w

    meta = {
        "dataset": DATASET,
        "hemisphere": SIDE,
        "source": "neuPrint public Cypher endpoint",
        "counts": {
            "ALPN": len(pn_ids),
            "KC": len(kc_ids),
            "MBON": len(mbon_ids),
            "edges_pn_kc": int((W_pn_kc > 0).sum()),
            "edges_kc_mbon": int((W_kc_mbon > 0).sum()),
        },
        "pn": [{"bodyId": b, "type": by_id[b]["type"],
                "glomerulus": (by_id[b]["type"] or "?").split("_")[0]} for b in pn_ids],
        "kc": [{"bodyId": b, "type": by_id[b]["type"]} for b in kc_ids],
        "mbon": [{"bodyId": b, "type": by_id[b]["type"],
                  "instance": by_id[b]["instance"]} for b in mbon_ids],
    }
    return W_pn_kc, W_kc_mbon, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refetch even if cached")
    args = ap.parse_args()

    if os.path.exists(NPZ) and not args.force:
        print(f"cache hit: {NPZ} (use --force to refetch)")
        return

    ensure(DATA)
    nodes, pn_kc, kc_mbon = fetch()
    W_pn_kc, W_kc_mbon, meta = build(nodes, pn_kc, kc_mbon)

    np.savez_compressed(NPZ, W_pn_kc=W_pn_kc, W_kc_mbon=W_kc_mbon)
    with open(META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    c = meta["counts"]
    print("\nsaved")
    print(f"  ALPN x KC   : {W_pn_kc.shape}  {c['edges_pn_kc']} edges")
    print(f"  KC x MBON   : {W_kc_mbon.shape}  {c['edges_kc_mbon']} edges")
    print(f"  {NPZ}")
    print(f"  {META}")


if __name__ == "__main__":
    main()
