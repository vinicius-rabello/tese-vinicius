import torch
import numpy as np
from torch.utils.data import Dataset
import os


# Seed offsets for the fixed per-sample noise of each split, so validation
# and test noise never coincide (only used when noise_std > 0).
NOISE_SEED_OFFSET = {'val': 1_000_000, 'test': 2_000_000}


class SuperResDataset(Dataset):
    def __init__(self, hr_files, downsample_factor=4, freq_threshold=None,
                 noise_std=0.0, noise_mode='fixed', noise_seed_offset=0):
        """
        Parameters
        ----------
        hr_files : list[str]
            List of yearly .npy files. Each file must have shape
            (N, 128, 128, 2).
        downsample_factor : int
            Downsampling factor used to generate LR inputs.
        noise_std : float
            Standard deviation (m/s) of white Gaussian observation noise added
            to the LR input after the low-pass filter; the HR target stays
            clean. 0 (the default) adds nothing and leaves the dataset exactly
            as before.
        noise_mode : 'fixed' or 'random'
            'fixed' (validation/test): the noise of sample i is drawn from its
            own generator seeded with noise_seed_offset + i, so every model and
            baseline sees the same noise on the same sample. 'random'
            (training): fresh noise on every access from torch's global RNG,
            which the run seed, the DataLoader generator and worker_init_fn
            control.
        noise_seed_offset : int
            Keeps the fixed noise of different splits independent (sample 0 of
            validation and sample 0 of test get different noise).
        """

        print("Loading datasets...")

        arrays = []
        total_samples = 0

        for file in hr_files:
            arr = np.load(file).astype(np.float32)

            print(f"{file}: {arr.shape}")

            if arr.ndim != 4:
                raise ValueError(
                    f"{file} should have shape (N, H, W, C), got {arr.shape}"
                )

            if arr.shape[-1] != 2:
                raise ValueError(
                    f"{file} should have 2 channels (u,v), got {arr.shape[-1]}"
                )

            arrays.append(arr)
            total_samples += arr.shape[0]

        self.hr_data = np.concatenate(arrays, axis=0)

        print(f"Loaded {total_samples} samples.")
        print(f"Dataset shape: {self.hr_data.shape}")

        self.downsample_factor = downsample_factor
        self.freq_threshold = freq_threshold
        if noise_mode not in ('fixed', 'random'):
            raise ValueError(f"noise_mode must be 'fixed' or 'random', got {noise_mode!r}")
        self.noise_std = float(noise_std)
        self.noise_mode = noise_mode
        self.noise_seed_offset = int(noise_seed_offset)

        N, H, W, C = self.hr_data.shape
        f = downsample_factor

        if H % f != 0 or W % f != 0:
            raise ValueError(
                f"Image size ({H}, {W}) must be divisible by {f}."
            )

        # Downsample by block averaging
        reshaped = self.hr_data.reshape(
            N,
            H // f, f,
            W // f, f,
            C
        )

        lr = reshaped.mean(axis=(2, 4))

        # Upsample back to original size
        lr = np.repeat(lr, f, axis=1)
        lr = np.repeat(lr, f, axis=2)

        self.lr_data = lr.astype(np.float32)

    def __len__(self):
        return len(self.hr_data)

    def apply_gaussian_low_pass_filter(self, tensor, sigma):
        C, H, W = tensor.shape
        freq_domain = torch.fft.fft2(tensor)
        freq_shifted = torch.fft.fftshift(freq_domain)

        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist_sq = (x - cx)**2 + (y - cy)**2
        
        mask = torch.exp(-dist_sq / (2 * (sigma**2)))
        
        freq_shifted_filtered = freq_shifted * mask.to(freq_shifted.device)
        freq_filtered = torch.fft.ifftshift(freq_shifted_filtered)
        filtered_tensor = torch.fft.ifft2(freq_filtered)
        
        return filtered_tensor.real
    

    def __getitem__(self, idx):

        lr = torch.from_numpy(self.lr_data[idx]).permute(2, 0, 1)
        hr = torch.from_numpy(self.hr_data[idx]).permute(2, 0, 1)

        if self.freq_threshold is not None:
            lr = self.apply_gaussian_low_pass_filter(lr, self.freq_threshold)

        # With noise_std = 0 nothing below runs and no RNG is touched.
        if self.noise_std > 0:
            if self.noise_mode == 'fixed':
                g = torch.Generator().manual_seed(self.noise_seed_offset + idx)
                noise = torch.randn(lr.shape, generator=g, dtype=lr.dtype)
            else:
                noise = torch.randn(lr.shape, dtype=lr.dtype)
            lr = lr + self.noise_std * noise

        return lr, hr
    


if __name__ == "__main__":
    dataset = SuperResDataset(
        hr_files=[
            f"./hycom_numpy/hycom_{year}.npy"
            for year in range(2001, 2020)
        ],
        downsample_factor=4,
        freq_threshold=10
    )

    print(f"Dataset size: {len(dataset)}")

    lr, hr = dataset[0]

    print("LR:", lr.shape)
    print("HR:", hr.shape)