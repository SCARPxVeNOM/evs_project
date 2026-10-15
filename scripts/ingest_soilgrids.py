"""Load real ISRIC SoilGrids soil classification for each hobli.

    python scripts/ingest_soilgrids.py --check     # report only
    python scripts/ingest_soilgrids.py             # write data/soil_wrb.csv
    python scripts/ingest_soilgrids.py --fetch     # re-query the ISRIC API first

Source
------
ISRIC SoilGrids 2.0, ``classification/query`` endpoint, sampled at each hobli
centroid. The response is the WRB reference soil group with a probability vector
over the most likely classes. Vendored in
``data/raw/isric_soilgrids_wrb_hoblis.json``.

    https://rest.isric.org/soilgrids/v2.0/classification/query

Why this does not set the infiltration rate
-------------------------------------------
It is tempting to drive ``infiltration_mm_per_hr`` from the soil class, and that
was tried. It makes the model worse, so it was not kept.

Across all 22 hoblis SoilGrids returns only two modal classes - Luvisols for 15,
Lixisols for 7 - and the modal probability is only 22% to 46%. Both are argic
(clay-enriched subsoil) groups with similar permeability, so a probability-weighted
rate collapses into a 12.3-18.6 mm/hr band: a spread of 6 mm/hr where the
texture-based estimates span 23. Since the rule engine screens recharge pits at
5 mm/hr, every hobli would clear the threshold and infiltration would stop
discriminating anything. Varthur and Sarjapura, which are clay and currently get
appropriately cautious sizing, would be treated as freely draining.

So the classification is recorded as what it honestly is - real, independent
evidence about the soil, at coarse spatial confidence - and used two ways:

  * reported alongside the assumed rate, so a reader can see both
  * as a **cross-check**: where SoilGrids puts meaningful probability on
    Vertisols (shrink-swell clay, which is what actually kills a percolation pit)
    but the assumed rate is fast, the hobli is flagged and a caveat is raised

That is a real use of real data without pretending a 27%-confidence point sample
outranks a texture estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW = DATA_DIR / "raw" / "isric_soilgrids_wrb_hoblis.json"
OUT = DATA_DIR / "soil_wrb.csv"

API = ("https://rest.isric.org/soilgrids/v2.0/classification/query"
       "?lon={lon}&lat={lat}&number_classes=5")
SOURCE_REF = "ISRIC-SoilGrids-2.0-WRB"

# A hobli is flagged when the soil model puts at least this much probability on
# shrink-swell clay while the assumed rate says the ground drains freely.
VERTISOL_FLAG_PROBABILITY = 0.12
FAST_INFILTRATION_MM_PER_HR = 20.0


class IngestError(RuntimeError):
    pass


def fetch(hoblis: list[dict], pause: float = 0.6) -> dict:
    """Re-query ISRIC for every hobli centroid. Slow; the vendored file is normal."""
    out: dict = {}
    for i, row in enumerate(hoblis, 1):
        url = API.format(lon=row["lon"], lat=row["lat"])
        last: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as fh:
                    data = json.load(fh)
                out[row["hobli"]] = {
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "wrb": data.get("wrb_class_name"),
                    "probs": data.get("wrb_class_probability"),
                    "current_soil": row["soil_type"],
                    "current_infil": float(row["infiltration_mm_per_hr"]),
                }
                print(f"  {i:2}/{len(hoblis)} {row['hobli']:26} {data.get('wrb_class_name')}")
                last = None
                break
            except Exception as exc:  # noqa: BLE001 - network shape varies
                last = exc
                time.sleep(3)
        if last is not None:
            raise IngestError(f"ISRIC query failed for {row['hobli']}: {last}")
        time.sleep(pause)
    return out


def read_raw(path: Path = RAW) -> dict:
    if not path.exists():
        raise IngestError(f"{path} is missing; run with --fetch to build it.")
    data = json.loads(path.read_text(encoding="utf-8"))
    broken = [k for k, v in data.items() if not v.get("wrb")]
    if broken:
        raise IngestError(f"no classification recorded for: {broken}")
    return data


def build(data: dict, hoblis: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in hoblis:
        name = row["hobli"]
        if name not in data:
            raise IngestError(f"{name} has no SoilGrids classification")
        record = data[name]
        probs = record["probs"]
        total = sum(p for _, p in probs) or 1
        mix = {n: p / total for n, p in probs}
        vertisol = mix.get("Vertisols", 0.0)
        assumed = float(row["infiltration_mm_per_hr"])

        disagrees = (
            vertisol >= VERTISOL_FLAG_PROBABILITY and assumed >= FAST_INFILTRATION_MM_PER_HR
        )
        rows.append(
            {
                "hobli": name,
                "wrb_class": record["wrb"],
                "wrb_confidence_pct": round(probs[0][1] / total * 100, 1),
                "wrb_class_mix": "; ".join(f"{n} {p / total:.0%}" for n, p in probs),
                "vertisol_probability_pct": round(vertisol * 100, 1),
                "assumed_infiltration_mm_per_hr": assumed,
                "infiltration_disputed": "yes" if disagrees else "no",
                "data_grade": "isric_soilgrids_point",
                "source_ref": SOURCE_REF,
            }
        )
    return rows


def write(rows: list[dict]) -> Path:
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return OUT


def check(rows: list[dict]) -> list[str]:
    problems: list[str] = []
    for row in rows:
        if not row["wrb_class"]:
            problems.append(f"{row['hobli']}: no WRB class")
        if not 0 <= row["wrb_confidence_pct"] <= 100:
            problems.append(f"{row['hobli']}: confidence {row['wrb_confidence_pct']} out of range")
        if not 0 <= row["vertisol_probability_pct"] <= 100:
            problems.append(f"{row['hobli']}: vertisol probability out of range")
    if len({r["hobli"] for r in rows}) != len(rows):
        problems.append("duplicate hobli rows")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    parser.add_argument("--fetch", action="store_true", help="re-query the ISRIC API first")
    args = parser.parse_args()

    with (DATA_DIR / "hoblis.csv").open(newline="", encoding="utf-8") as fh:
        hoblis = list(csv.DictReader(fh))

    if args.fetch:
        print(f"querying ISRIC SoilGrids for {len(hoblis)} hobli centroids:")
        data = fetch(hoblis)
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {RAW.relative_to(ROOT)}")
    else:
        data = read_raw()

    rows = build(data, hoblis)

    print(f"\n{'hobli':26} {'WRB class':12} {'conf':>5} {'vert':>6} {'assumed':>8}  check")
    for row in rows:
        mark = "DISPUTED" if row["infiltration_disputed"] == "yes" else "ok"
        print(
            f"{row['hobli']:26} {row['wrb_class']:12} {row['wrb_confidence_pct']:4.0f}% "
            f"{row['vertisol_probability_pct']:5.1f}% {row['assumed_infiltration_mm_per_hr']:7.0f}  {mark}"
        )

    classes: dict[str, int] = {}
    for row in rows:
        classes[row["wrb_class"]] = classes.get(row["wrb_class"], 0) + 1
    disputed = [r["hobli"] for r in rows if r["infiltration_disputed"] == "yes"]
    print(f"\nmodal classes: {classes}")
    print(
        f"confidence range: {min(r['wrb_confidence_pct'] for r in rows):.0f}%"
        f"-{max(r['wrb_confidence_pct'] for r in rows):.0f}% - too coarse to set the rate from, "
        "which is why it only cross-checks it"
    )
    print(f"{len(disputed)} hobli(s) flagged where clay probability conflicts with the assumed rate:")
    for name in disputed:
        print(f"  - {name}")

    problems = check(rows)
    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nchecks passed")

    if args.check:
        print("--check given, nothing written")
        return
    print(f"wrote {write(rows).relative_to(ROOT)} ({len(rows)} hoblis)")


if __name__ == "__main__":
    main()
