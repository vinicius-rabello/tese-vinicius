import sys
import os
import shutil
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.sr_tiny_dataset import SRTinyDataset

# --- Configuration ---
HR_FILES = ['data/100/window_2003.npy']  # Update as needed
DOWNSAMPLE_FACTOR = 1
OUTPUT_DIR = 'data/sr_tiny_dataset/train'

# --- Reset output folder ---
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

dataset = SRTinyDataset(hr_files=HR_FILES, downsample_factor=DOWNSAMPLE_FACTOR)
print(f"Dataset length: {len(dataset)}")

# --- Save each item ---
for idx in range(len(dataset)):
    _, hr_tensor = dataset[idx]
    hr = hr_tensor.numpy()  # shape: (2, H, W)

    for ch_idx, ch_name in enumerate(['u', 'v']):
        plt.imsave(os.path.join(OUTPUT_DIR, f'{idx:05d}_hr_{ch_name}.png'), hr[ch_idx], cmap='gray')

    if (idx + 1) % 100 == 0:
        print(f"Saved {idx + 1}/{len(dataset)} items...")

print(f"Done! All images saved to '{OUTPUT_DIR}'")