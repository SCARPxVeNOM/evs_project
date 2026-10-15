"""Derive the monthly rainfall shape and the dry-year scenario from real records.

    python scripts/ingest_city_monthly.py --check    # report only
    python scripts/ingest_city_monthly.py            # patch data/parameters.json

Source
------
``data/raw/imd_bengaluru_monthly_1901_2025.csv`` - Bengaluru monthly rainfall for
every year from 1901 to 2025, plus IMD's published long-period averages for
1981-2010 and 1991-2020, as collated by OpenCity (public domain) from IMD records:

    https://data.opencity.in/dataset/bengaluru-rainfall

What this script writes
-----------------------
Three things go into ``data/parameters.json``:

``monthly_rainfall_normal_mm``
    IMD's published 1991-2020 monthly normals, in millimetres. 1991-2020 is the
    current WMO standard reference period.

``monthly_rainfall_shares``
    The same normals expressed as a share of the annual total. This is what
    distributes a hobli's measured season totals across the months inside each
    season, so the shape is now anchored in 30 years of gauge records rather than
    a secondary table. The figures it replaces were wrong by a wide margin in the
    months that matter most for first filling a tank - April was understated by
    27 mm and June by 30 mm.

``dry_year``
    The mean monthly pattern of the eight driest years since 1991, as a ratio to
    the 1991-2020 normal. This is deliberately *not* a flat percentage. A dry year
    in Bengaluru does not scale down evenly: it is the September peak that fails,
    dropping to about 39% of normal, while June holds near 100%. Scaling every
    month by one factor would hide exactly the risk a household needs to see,
    because September is when a tank should be filling for the dry season ahead.

The published LPA is used in preference to recomputing it from the yearly rows.
The two differ by about 3% (1074 mm against 1043 mm), which says the LPA row comes
from IMD's own normals rather than from this file's own series; ``--check`` prints
the comparison so the gap stays visible rather than being quietly averaged away.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW = DATA_DIR / "raw" / "imd_bengaluru_monthly_1901_2025.csv"
PARAMS = DATA_DIR / "parameters.json"

# Column spellings as they appear in the source file.
SOURCE_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "June",
                 "July", "Aug", "Sept", "Oct", "Nov", "Dec"]

REFERENCE_PERIOD = "LPA (1991-2020)"
DRY_YEAR_COUNT = 8
DRY_YEAR_FROM = 1991
SOURCE_REF = "IMD-Bengaluru-monthly-1901-2025;OpenCity"


class IngestError(RuntimeError):
    pass


def read_series(path: Path = RAW) -> tuple[dict[int, list[float]], dict[str, list[float]]]:
    """Return ``(yearly, published_lpa)`` monthly series in millimetres."""
    if not path.exists():
        raise IngestError(f"{path} is missing; see this script's docstring for the URL.")

    yearly: dict[int, list[float]] = {}
    lpa: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("Year") or "").strip()
            try:
                values = [float((row[m] or "").strip()) for m in SOURCE_MONTHS]
            except (ValueError, KeyError):
                continue
            if label.startswith("LPA"):
                lpa[label] = values
            elif label.isdigit():
                yearly[int(label)] = values

    if REFERENCE_PERIOD not in lpa:
        raise IngestError(f"{REFERENCE_PERIOD!r} row not found in {path.name}")
    if len(yearly) < 100:
        raise IngestError(f"expected a century of records, found {len(yearly)} years")
    return yearly, lpa


def dry_year_ratios(
    yearly: dict[int, list[float]], normal: list[float]
) -> tuple[dict[int, float], list[int], float]:
    """Monthly ratio to normal for the mean of the driest recent years."""
    candidates = sorted(
        ((sum(v), y) for y, v in yearly.items() if y >= DRY_YEAR_FROM),
    )[:DRY_YEAR_COUNT]
    years = sorted(y for _, y in candidates)
    if len(years) < DRY_YEAR_COUNT:
        raise IngestError(f"only {len(years)} years available since {DRY_YEAR_FROM}")

    mean_monthly = [statistics.mean(yearly[y][i] for y in years) for i in range(12)]
    ratios = {
        i + 1: round(mean_monthly[i] / normal[i], 4) if normal[i] > 0 else 1.0
        for i in range(12)
    }
    annual_fraction = sum(mean_monthly) / sum(normal)
    return ratios, years, annual_fraction


def percentile(data: list[float], q: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def build(yearly: dict[int, list[float]], lpa: dict[str, list[float]]) -> dict:
    normal = lpa[REFERENCE_PERIOD]
    total = sum(normal)
    ratios, years, annual_fraction = dry_year_ratios(yearly, normal)
    annual_totals = [sum(v) for v in yearly.values()]

    return {
        "monthly_rainfall_normal_mm": {
            "_note": (
                f"IMD published monthly normals for Bengaluru, {REFERENCE_PERIOD.strip('LPA ()')} "
                f"(the current WMO reference period), totalling {total:.1f} mm. "
                f"Source: {SOURCE_REF}."
            ),
            **{str(i + 1): normal[i] for i in range(12)},
            "annual_total_mm": round(total, 1),
        },
        "monthly_rainfall_shares": {
            "_note": (
                "Share of annual rainfall by month, from the IMD 1991-2020 Bengaluru monthly "
                "normals. Used to distribute a hobli's measured KSNDMC season totals across the "
                f"months inside each season. Source: {SOURCE_REF}."
            ),
            **{str(i + 1): round(normal[i] / total, 5) for i in range(12)},
        },
        "dry_year": {
            "_note": (
                f"Mean monthly rainfall of the {DRY_YEAR_COUNT} driest years since {DRY_YEAR_FROM}, "
                "as a ratio to the 1991-2020 normal. Not a flat percentage: a dry year in "
                "Bengaluru fails at the September peak (about 39% of normal) while June holds "
                "near normal, so scaling every month alike would hide the real risk. "
                f"Years used: {', '.join(str(y) for y in years)}. Source: {SOURCE_REF}."
            ),
            "years_used": years,
            "annual_fraction_of_normal": round(annual_fraction, 4),
            "monthly_ratio": {str(m): r for m, r in ratios.items()},
            "record_years": len(yearly),
            "annual_percentiles_mm": {
                "p05": round(percentile(annual_totals, 0.05), 1),
                "p10": round(percentile(annual_totals, 0.10), 1),
                "p25": round(percentile(annual_totals, 0.25), 1),
                "p50": round(percentile(annual_totals, 0.50), 1),
                "p75": round(percentile(annual_totals, 0.75), 1),
                "p90": round(percentile(annual_totals, 0.90), 1),
            },
        },
    }


def report(yearly: dict[int, list[float]], lpa: dict[str, list[float]], blocks: dict) -> None:
    normal = lpa[REFERENCE_PERIOD]
    params = json.loads(PARAMS.read_text(encoding="utf-8"))
    old = params["monthly_rainfall_shares"]
    old_total = 961.6  # the annual the old shares were taken against

    window = [yearly[y] for y in range(1991, 2021) if y in yearly]
    recomputed = [statistics.mean(v[i] for v in window) for i in range(12)]
    print(f"records read: {len(yearly)} years ({min(yearly)}-{max(yearly)})")
    print(
        f"published {REFERENCE_PERIOD} total {sum(normal):.1f} mm; "
        f"recomputed from the same file's {len(window)} yearly rows {sum(recomputed):.1f} mm "
        f"(max monthly gap {max(abs(a - b) for a, b in zip(recomputed, normal)):.1f} mm)"
    )

    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    dry = blocks["dry_year"]["monthly_ratio"]
    print(f"\n{'month':6}{'old mm':>9}{'real mm':>9}{'change':>9}{'dry yr':>9}{'ratio':>8}")
    for i, name in enumerate(names):
        om = float(old[str(i + 1)]) * old_total
        print(
            f"{name:6}{om:9.1f}{normal[i]:9.1f}{normal[i] - om:+9.1f}"
            f"{normal[i] * float(dry[str(i + 1)]):9.1f}{float(dry[str(i + 1)]):8.2f}"
        )

    print(
        f"\ndry-year scenario: {dry_label(blocks)} "
        f"({blocks['dry_year']['annual_fraction_of_normal']:.0%} of normal overall)"
    )
    pcts = blocks["dry_year"]["annual_percentiles_mm"]
    print("annual totals across the full record: " + "  ".join(f"{k.upper()}={v:.0f}" for k, v in pcts.items()))


def dry_label(blocks: dict) -> str:
    years = blocks["dry_year"]["years_used"]
    return f"mean of {len(years)} driest years since {DRY_YEAR_FROM} ({years[0]}-{years[-1]})"


def check(blocks: dict) -> list[str]:
    problems: list[str] = []

    shares = {k: v for k, v in blocks["monthly_rainfall_shares"].items() if k.isdigit()}
    if abs(sum(shares.values()) - 1.0) > 0.001:
        problems.append(f"monthly shares sum to {sum(shares.values()):.5f}, not 1.0")
    if len(shares) != 12:
        problems.append(f"expected 12 monthly shares, got {len(shares)}")

    normal = {k: v for k, v in blocks["monthly_rainfall_normal_mm"].items() if k.isdigit()}
    stated = blocks["monthly_rainfall_normal_mm"]["annual_total_mm"]
    if abs(sum(normal.values()) - stated) > 0.5:
        problems.append(f"monthly normals sum to {sum(normal.values()):.1f}, stated {stated}")
    if not 800 <= stated <= 1300:
        problems.append(f"annual normal {stated} mm is implausible for Bengaluru")

    dry = blocks["dry_year"]
    ratios = {int(k): float(v) for k, v in dry["monthly_ratio"].items()}
    if len(ratios) != 12:
        problems.append(f"expected 12 dry-year ratios, got {len(ratios)}")
    for month, ratio in sorted(ratios.items()):
        if not 0.0 <= ratio <= 2.0:
            problems.append(f"dry-year ratio for month {month} is {ratio}, outside 0-2")
    if not 0.4 <= dry["annual_fraction_of_normal"] <= 0.95:
        problems.append(
            f"dry year is {dry['annual_fraction_of_normal']:.0%} of normal, which is not a dry year"
        )
    # the whole point: the pattern must not be a flat scaling
    if max(ratios.values()) - min(ratios.values()) < 0.25:
        problems.append("dry-year ratios are nearly flat; the monthly pattern was lost")

    return problems


def write(blocks: dict) -> Path:
    params = json.loads(PARAMS.read_text(encoding="utf-8"))
    params.update(blocks)
    PARAMS.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return PARAMS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report and validate, write nothing")
    args = parser.parse_args()

    yearly, lpa = read_series()
    blocks = build(yearly, lpa)
    report(yearly, lpa, blocks)

    problems = check(blocks)
    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nchecks passed")

    if args.check:
        print("--check given, nothing written")
        return

    print(f"wrote {write(blocks).relative_to(ROOT)}")
    print("now re-run: python scripts/ingest_ksndmc.py && python -m ml.train --regenerate")


if __name__ == "__main__":
    main()
