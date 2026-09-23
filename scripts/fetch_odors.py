"""Download the three DoOR 2.0 tables the odour pipeline needs.

DoOR ships as an R package; these are the plain CSVs inside it. They are not
committed (they belong upstream and are large-ish), so a fresh clone needs this
before build_odor_map.py will run.

Source: https://github.com/ropensci/DoOR.data
Paper:  Munch & Galizia, Scientific Reports 6:21841 (2016)
"""
import argparse
import os
import sys

import requests

from flymb.paths import DATA, ensure

RAW = "https://raw.githubusercontent.com/ropensci/DoOR.data/master/data"
FILES = {
    "door_response_matrix.csv": "691 chemicals x 78 receptors, consensus responses",
    "odor.csv": "chemical names and classes",
    "door_mappings.csv": "receptor -> glomerulus",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="redownload even if present")
    args = ap.parse_args()

    ensure(DATA)
    for name, what in FILES.items():
        dest = os.path.join(DATA, name)
        if os.path.exists(dest) and not args.force:
            print(f"  have  {name:<28} {os.path.getsize(dest)/1024:>7.0f} KB")
            continue
        url = f"{RAW}/{name}"
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - report and stop, don't half-finish
            print(f"  FAIL  {name}: {exc}", file=sys.stderr)
            print(f"        fetch it by hand from {url}", file=sys.stderr)
            return 1
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  got   {name:<28} {len(r.content)/1024:>7.0f} KB   {what}")
    print(f"\n-> {DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
