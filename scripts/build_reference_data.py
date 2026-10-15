"""Build the derived reference datasets from the curated tables in ``data/``.

Right now that means one file: ``data/rainfall_monthly.csv``, the month-by-month
rainfall and rainy-day counts per hobli. It is derived, not measured - each
hobli's annual total from ``data/hoblis.csv`` is distributed across the year
using the IMD Bengaluru City monthly shares in ``data/parameters.json``.

Run it after editing ``data/hoblis.csv``::

    python scripts/build_reference_data.py

The shipped table is no longer built this way. ``scripts/ingest_ksndmc.py`` now
fills the same file from real KSNDMC gauge normals, which carry their own
per-hobli seasonality instead of one shared curve, so this script refuses to run
over measured data unless given ``--force``. It is kept because it is the
fallback for a hobli KSNDMC does not gauge, and because the derived table is what
the ingestion is measured against.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def build_rainfall_monthly() -> Path:
    params = json.loads((DATA_DIR / "parameters.json").read_text(encoding="utf-8"))
    rain_shares = {int(k): v for k, v in params["monthly_rainfall_shares"].items() if k.isdigit()}
    day_shares = {int(k): v for k, v in params["monthly_rainy_day_shares"].items() if k.isdigit()}

    with (DATA_DIR / "hoblis.csv").open(newline="", encoding="utf-8") as fh:
        hoblis = list(csv.DictReader(fh))

    rows = []
    for hobli in hoblis:
        annual_mm = float(hobli["annual_rainfall_mm"])
        annual_days = float(hobli["annual_rainy_days"])
        for month in range(1, 13):
            rows.append(
                {
                    "hobli": hobli["hobli"],
                    "taluk": hobli["taluk"],
                    "month": month,
                    "month_name": MONTH_NAMES[month - 1],
                    "rainfall_mm": round(annual_mm * rain_shares[month], 1),
                    "rainy_days": round(annual_days * day_shares[month], 1),
                    "data_grade": "derived_from_annual",
                    "source_ref": hobli["source_ref"],
                }
            )

    out = DATA_DIR / "rainfall_monthly.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def _currently_measured() -> str | None:
    """The grade of the rainfall table on disk, if it is better than derived."""
    path = DATA_DIR / "rainfall_monthly.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            grade = (row.get("data_grade") or "").strip()
            return grade if grade.startswith("ksndmc") else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite even if the table on disk holds measured KSNDMC records",
    )
    args = parser.parse_args()

    grade = _currently_measured()
    if grade and not args.force:
        raise SystemExit(
            f"data/rainfall_monthly.csv currently holds {grade} records from real KSNDMC\n"
            "gauges. Rebuilding it from annual totals would replace measured seasonality\n"
            "with one shared IMD curve - a downgrade.\n\n"
            "Run scripts/ingest_ksndmc.py to refresh from the KSNDMC source, or pass\n"
            "--force if you really do want the derived table back."
        )

    out = build_rainfall_monthly()
    rows = sum(1 for _ in out.open(encoding="utf-8")) - 1
    print(f"wrote {out} ({rows} rows)")


if __name__ == "__main__":
    main()
