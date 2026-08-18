from pathlib import Path
from netCDF4 import Dataset
import numpy as np

WINDOW = 512
TARGET_SIZE = 128
BLOCK_SIZE = WINDOW // TARGET_SIZE

# Placeholder window location (upper-left corner)
ROW = 512
COL = 794

input_root = Path("/home/lcadame/Vinicius_Thesis/Data/hycom/100")
output_root = Path("/home/lcadame/Vinicius_Thesis/Vinicius/hycom_numpy")
output_root.mkdir(exist_ok=True)

for year in range(2001, 2025):

    print(f"Processing {year}...")

    year_samples = []
    month_days = []

    for month in range(1, 13):

        path = input_root / str(year) / f"hycom_gom_{year}-{month:02d}.nc"

        if not path.exists():
            print(f"Missing {path}")
            continue

        with Dataset(path) as ds:

            num_days = ds["u"].shape[0]
            month_days.append(num_days)

            u = ds["u"][:].filled(0.0)
            v = ds["v"][:].filled(0.0)

            data = np.stack((u, v), axis=-1).astype(np.float32)

            # data.shape = (days, lat, lon, 2)

            patches = data[
                :,
                ROW:ROW+WINDOW,
                COL:COL+WINDOW,
                :
            ]

            days = patches.shape[0]
            reshaped = patches.reshape(
                days, 
                TARGET_SIZE, BLOCK_SIZE, 
                TARGET_SIZE, BLOCK_SIZE, 
                2
            )

            downsampled_patches = reshaped.mean(axis=(2, 4))
            year_samples.append(downsampled_patches)

    if not year_samples:
        continue

    year_samples = np.concatenate(year_samples, axis=0)

    print(f"Year {year}: {year_samples.shape}")
    print(f"{year}: {month_days} -> total={sum(month_days)}")

    np.save(
        output_root / f"hycom_{year}.npy",
        year_samples
    )