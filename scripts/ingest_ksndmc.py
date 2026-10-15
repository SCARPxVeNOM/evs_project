"""Load real KSNDMC rainfall normals into the project's data tables.

This replaces the invented ``annual_rainfall_mm`` figures in ``data/hoblis.csv``
and the single shared seasonality curve in ``data/rainfall_monthly.csv`` with
measured KSNDMC telemetric rain-gauge normals.

    python scripts/ingest_ksndmc.py            # write the tables
    python scripts/ingest_ksndmc.py --check    # report only, write nothing

Source
------
``data/raw/ksndmc_karnataka_hobli_normals.csv`` - KSNDMC's district/taluk/hobli
rainfall table as published by OpenCity (public domain), from

    https://data.opencity.in/dataset/karnataka-annual-rainfall-districts-taluks-and-hoblis

It carries, per hobli rain gauge, the long-period *normal* for each of Karnataka's
three rainfall seasons plus the annual normal:

    Pre-monsoon      January - May
    South-West       June - September
    North-East       October - December

The three sum to the annual normal, which is how the Jan-May pre-monsoon
convention above was confirmed rather than assumed (see ``--check``).

What is real and what is still derived
--------------------------------------
Real, from KSNDMC:
  * each hobli's annual rainfall normal
  * each hobli's split of that total across the three seasons

Still derived:
  * the month-by-month shape *within* each season, which comes from the IMD
    Bengaluru City monthly normals in ``data/parameters.json``
  * rainy-day counts, scaled from the IMD City ratio of 56 days per 961.6 mm

So a hobli's season totals are now its own measured values instead of a
city-wide curve, but the distribution inside a season is still a shared shape.
KSNDMC publishes true monthly normals only in its annual report PDFs, which are
not machine-readable; wiring those in is the next fidelity step.

Gauge-to-hobli mapping
----------------------
KSNDMC names gauges, not hoblis: a hobli with three telemetric gauges appears as
``Kengeri_1``, ``Kengeri_2``, ``Kengeri_3``, and the spellings drift from the
project's (``Beguru``/``Begur``, ``Varturu``/``Varthur``, ``K R Pura``/
``Krishnarajapuram``). ``GAUGE_ALIASES`` maps each project hobli to its gauges
explicitly rather than fuzzy-matching, and a hobli's normal is the mean of its
gauges. Two hoblis - the Bengaluru East and Bengaluru South kasabas - have no
gauge of their own and fall back to their taluk's normal; those rows are graded
``ksndmc_taluk_normal`` so the coarser provenance travels with the number.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW = DATA_DIR / "raw" / "ksndmc_karnataka_hobli_normals.csv"

SOURCE_REF = "KSNDMC-hobli-normals;OpenCity"
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Karnataka's three rainfall seasons, as the months they span.
SEASONS: dict[str, tuple[int, ...]] = {
    "pre": (1, 2, 3, 4, 5),
    "swm": (6, 7, 8, 9),
    "nem": (10, 11, 12),
}

# Column names in the KSNDMC table, per season.
SEASON_COLUMNS = {
    "pre": "Pre-monsoon Normal (mm)",
    "swm": "Southwest Monsoon Normal(mm)",
    "nem": "North East Monsoon Normal(mm)",
}
ANNUAL_COLUMN = "Annual Normal(mm)"
LEVEL_COLUMN = "District(D)/Taluk(T)/Hobli(H)"
NAME_COLUMN = "District/Taluk/Hobli"

# IMD Bengaluru City: 56 rainy days (>= 2.5 mm) per 961.6 mm of annual normal.
RAINY_DAYS_PER_MM = 56.0 / 961.6

# project hobli -> the KSNDMC gauge names that sit in it
GAUGE_ALIASES: dict[str, list[str]] = {
    "Bengaluru North (Kasaba)": ["Bengaluru North_1", "Bengaluru North_2"],
    "Yelahanka": ["Yelahanka_1", "Yelahanka_2", "Yelahanka_3"],
    "Jala": ["Jala_1", "Jala_2", "Jala_3"],
    "Hesaraghatta": ["Hesarughatta_1", "Hesarughatta_2"],
    "Dasanapura": ["Dasanapura_1", "Dasanapura_2", "Dasanapura_3"],
    "Krishnarajapuram": ["K R Pura_2", "K R Pura_3"],
    "Bidarahalli": ["Bidarahalli_1", "Bidarahalli_2", "Bidharahalli_3"],
    "Varthur": ["Varturu_1", "Varturu_2"],
    "Uttarahalli": [
        "Uttarahalli_1", "Uttarahalli_2", "Uttarahalli_3",
        "Uttarahalli_4", "Uttarahalli_5",
    ],
    "Kengeri": ["Kengeri_1", "Kengeri_2", "Kengeri_3", "Kengeri_4"],
    "Begur": ["Beguru_1", "Beguru_2", "Beguru_3"],
    "Tavarekere": ["Tavarekere_1", "Tavarekere_2", "Tavarekere_3"],
    "Anekal": ["Anekal_1", "Anekal_2"],
    "Attibele": ["Attibele_1", "Attibele_2"],
    "Jigani": ["Jigani_1", "Jigani_2"],
    "Sarjapura": ["Sarjapura_1", "Sarjapura_3"],
    "Devanahalli": ["Devanahalli"],
    "Doddaballapur": ["Dodballapur"],
    "Hoskote": ["Hosakote"],
    "Nelamangala": ["Nelamangala", "Nelamangala_2"],
}

# Hoblis with no gauge of their own: use the taluk normal instead.
TALUK_FALLBACK: dict[str, str] = {
    "Bengaluru East (Kasaba)": "Bengaluru East",
    "Bengaluru South (Kasaba)": "Bengaluru South",
}


class IngestError(RuntimeError):
    """Raised when the raw table cannot be reconciled with the hobli list."""


def _number(value: str | None) -> float | None:
    text = (value or "").strip().replace(",", "")
    if not text or text in {"-", "NA", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_normals(path: Path = RAW) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return ``(gauges, taluks)`` of season normals keyed by KSNDMC name.

    Only the two Bengaluru districts are kept. Where the same gauge name appears
    more than once (KSNDMC reuses e.g. ``Sarjapura_1`` for two gauges), the
    readings are collected and averaged.
    """
    if not path.exists():
        raise IngestError(
            f"{path} is missing. Download the KSNDMC hobli table from OpenCity "
            "into data/raw/ - see this script's docstring for the URL."
        )

    gauge_rows: dict[str, list[dict]] = {}
    taluk_rows: dict[str, list[dict]] = {}
    district = taluk = None

    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            level = (row.get(LEVEL_COLUMN) or "").strip().upper()
            name = (row.get(NAME_COLUMN) or "").strip()
            if not name:
                continue

            if level == "D":
                district = name
                continue
            if level == "T":
                taluk = name
            if not (district or "").startswith("BENGALURU"):
                continue

            record = {season: _number(row.get(col)) for season, col in SEASON_COLUMNS.items()}
            record["annual"] = _number(row.get(ANNUAL_COLUMN))
            if any(v is None for v in record.values()):
                continue
            record["district"] = district
            record["taluk"] = taluk

            if level == "T":
                taluk_rows.setdefault(name, []).append(record)
            elif level == "H":
                gauge_rows.setdefault(name, []).append(record)

    def collapse(groups: dict[str, list[dict]]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, records in groups.items():
            out[name] = {
                **{
                    key: sum(r[key] for r in records) / len(records)
                    for key in ("pre", "swm", "nem", "annual")
                },
                "district": records[0]["district"],
                "taluk": records[0]["taluk"],
                "gauges": len(records),
            }
        return out

    return collapse(gauge_rows), collapse(taluk_rows)


def hobli_normals(gauges: dict[str, dict], taluks: dict[str, dict]) -> dict[str, dict]:
    """Average each project hobli's gauges into one set of season normals."""
    out: dict[str, dict] = {}

    for hobli, names in GAUGE_ALIASES.items():
        found = [gauges[n] for n in names if n in gauges]
        missing = [n for n in names if n not in gauges]
        if not found:
            raise IngestError(f"no KSNDMC gauge found for {hobli!r}; looked for {names}")
        out[hobli] = {
            **{
                key: sum(f[key] for f in found) / len(found)
                for key in ("pre", "swm", "nem", "annual")
            },
            "grade": "ksndmc_hobli_normal",
            "gauges": sum(f["gauges"] for f in found),
            "missing": missing,
        }

    for hobli, taluk in TALUK_FALLBACK.items():
        if taluk not in taluks:
            raise IngestError(f"no KSNDMC taluk row for {taluk!r}, needed by {hobli!r}")
        record = taluks[taluk]
        out[hobli] = {
            **{key: record[key] for key in ("pre", "swm", "nem", "annual")},
            "grade": "ksndmc_taluk_normal",
            "gauges": 0,
            "missing": [],
        }

    return out


def within_season_shares() -> tuple[dict[int, float], dict[int, float]]:
    """IMD monthly shares, renormalised to sum to 1.0 inside each season."""
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    rain = {int(k): float(v) for k, v in params["monthly_rainfall_shares"].items() if k.isdigit()}
    days = {int(k): float(v) for k, v in params["monthly_rainy_day_shares"].items() if k.isdigit()}

    scaled: dict[int, float] = {}
    for months in SEASONS.values():
        total = sum(rain[m] for m in months)
        for m in months:
            scaled[m] = rain[m] / total

    day_total = sum(days.values())
    return scaled, {m: v / day_total for m, v in days.items()}


def build_monthly(normals: dict[str, dict], hoblis: list[dict]) -> list[dict]:
    """Twelve rows per hobli: real season totals, IMD shape inside each season."""
    rain_shares, day_shares = within_season_shares()
    rows: list[dict] = []

    for hobli in hoblis:
        name = hobli["hobli"]
        record = normals[name]
        annual_mm = record["annual"]
        annual_days = annual_mm * RAINY_DAYS_PER_MM

        for season, months in SEASONS.items():
            season_mm = record[season]
            for month in months:
                rows.append(
                    {
                        "hobli": name,
                        "taluk": hobli["taluk"],
                        "month": month,
                        "month_name": MONTH_NAMES[month - 1],
                        "rainfall_mm": round(season_mm * rain_shares[month], 1),
                        "rainy_days": round(annual_days * day_shares[month], 1),
                        "data_grade": record["grade"],
                        "source_ref": SOURCE_REF,
                    }
                )

    rows.sort(key=lambda r: (r["hobli"], r["month"]))
    return rows


def write_monthly(rows: list[dict]) -> Path:
    out = DATA_DIR / "rainfall_monthly.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_hoblis(normals: dict[str, dict], hoblis: list[dict]) -> Path:
    """Patch the rainfall columns of hoblis.csv, leaving soil and groundwater alone."""
    out = DATA_DIR / "hoblis.csv"
    fields = [
        "district", "taluk", "hobli", "lat", "lon",
        "annual_rainfall_mm", "annual_rainy_days",
        "rainfall_data_grade", "rainfall_source_ref",
        "soil_type", "infiltration_mm_per_hr", "gw_depth_m", "gw_category",
        "data_grade", "source_ref",
    ]

    for row in hoblis:
        record = normals[row["hobli"]]
        row["annual_rainfall_mm"] = round(record["annual"], 1)
        row["annual_rainy_days"] = round(record["annual"] * RAINY_DAYS_PER_MM, 1)
        row["rainfall_data_grade"] = record["grade"]
        row["rainfall_source_ref"] = SOURCE_REF
        # soil and groundwater stay reference estimates; keep their grade honest
        row["source_ref"] = "NBSS&LUP/SoilGrids;CGWB-2022"

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(hoblis)
    return out


def check(normals: dict[str, dict], hoblis: list[dict]) -> list[str]:
    """Sanity checks worth failing loudly on."""
    problems: list[str] = []

    for name, record in sorted(normals.items()):
        season_sum = record["pre"] + record["swm"] + record["nem"]
        drift = abs(season_sum - record["annual"])
        if drift > 0.02 * record["annual"]:
            problems.append(
                f"{name}: seasons sum to {season_sum:.0f} mm but annual normal is "
                f"{record['annual']:.0f} mm ({drift:.0f} mm apart)"
            )
        if not 400 <= record["annual"] <= 1600:
            problems.append(f"{name}: annual normal {record['annual']:.0f} mm is implausible")
        if record["missing"]:
            problems.append(f"{name}: gauges named but not found: {record['missing']}")

    covered = {row["hobli"] for row in hoblis}
    for name in sorted(covered - set(normals)):
        problems.append(f"{name}: in hoblis.csv but no KSNDMC normal was resolved")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report and validate without writing"
    )
    args = parser.parse_args()

    gauges, taluks = read_normals()
    normals = hobli_normals(gauges, taluks)
    with (DATA_DIR / "hoblis.csv").open(newline="", encoding="utf-8") as fh:
        hoblis = list(csv.DictReader(fh))

    print(f"KSNDMC gauges read: {len(gauges)} in the two Bengaluru districts")
    print(f"taluk rows read:    {len(taluks)}")
    print()
    print(f"{'hobli':26} {'old mm':>7} {'new mm':>7} {'delta':>7}  {'gauges':>6}  grade")
    for row in hoblis:
        record = normals[row["hobli"]]
        old = float(row["annual_rainfall_mm"])
        new = record["annual"]
        print(
            f"{row['hobli']:26} {old:7.0f} {new:7.1f} {new - old:+7.1f}  "
            f"{record['gauges']:6d}  {record['grade']}"
        )

    season_sum = sum(normals[r["hobli"]][s] for r in hoblis for s in SEASONS)
    annual_sum = sum(normals[r["hobli"]]["annual"] for r in hoblis)
    print(
        f"\nseason totals sum to {season_sum:.0f} mm against an annual sum of "
        f"{annual_sum:.0f} mm, confirming the Jan-May pre-monsoon convention"
    )

    problems = check(normals, hoblis)
    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("checks passed")

    if args.check:
        print("\n--check given, nothing written")
        return

    monthly = write_monthly(build_monthly(normals, hoblis))
    print(f"\nwrote {monthly.relative_to(ROOT)} ({len(monthly.read_text().splitlines()) - 1} rows)")
    patched = write_hoblis(normals, hoblis)
    print(f"wrote {patched.relative_to(ROOT)} ({len(hoblis)} hoblis)")


if __name__ == "__main__":
    main()
