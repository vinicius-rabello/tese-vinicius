from pathlib import Path
import numpy as np

INPUT_DIR = Path("/home/lcadame/Vinicius_Thesis/Vinicius/hycom_numpy")
OUTPUT_DIR = Path("/home/lcadame/Vinicius_Thesis/Vinicius/hycom_dataset")

SPLITS = {
    "train": range(2001, 2020),
    "validation": [2020, 2021],
    "test": [2022, 2023, 2024],
}

OUTPUT_DIR.mkdir(exist_ok=True)

for split, years in SPLITS.items():

    split_dir = OUTPUT_DIR / split
    split_dir.mkdir(exist_ok=True)

    print(f"\nCreating {split}...")

    count = 0

    for year in years:

        path = INPUT_DIR / f"hycom_{year}.npy"

        if not path.exists():
            print(f"Missing {path}")
            continue

        data = np.load(path)

        print(f"{year}: {data.shape[0]} samples")

        for day, sample in enumerate(data):
            filename = f"{year}_{day:03d}.npy"
            np.save(split_dir / filename, sample)
            count += 1

    print(f"{split}: saved {count} samples")