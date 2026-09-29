"""Writes evaluation/test_dates.csv: the date of every test sample, in the
order the .npy files store them (create_dataset.py concatenates the monthly
netCDF files' time steps). The block bootstrap needs these dates.

    python scripts/make_test_dates.py
"""

import csv
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

ROOT = Path("/home/lcadame/Vinicius_Thesis/Data/hycom/100")
TEST_YEARS = (2022, 2023, 2024)
OUT = Path(__file__).resolve().parent.parent / "evaluation" / "test_dates.csv"


def main():
    rows, index = [], 0
    for year in TEST_YEARS:
        dates = []
        for month in range(1, 13):
            path = ROOT / str(year) / f"hycom_gom_{year}-{month:02d}.nc"
            if not path.exists():
                continue
            with Dataset(path) as ds:
                t = ds["MT"]
                dates += [str(d)[:10] for d in num2date(t[:], t.units, t.calendar)]
        n = np.load(f"hycom_numpy/hycom_{year}.npy", mmap_mode="r").shape[0]
        if len(dates) != n:
            raise RuntimeError(f"{year}: {len(dates)} dates but {n} samples in hycom_{year}.npy")
        for d in dates:
            rows.append({'index': index, 'date': d})
            index += 1
    with open(OUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'date'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} dates to {OUT} ({rows[0]['date']} .. {rows[-1]['date']})")


if __name__ == "__main__":
    main()
