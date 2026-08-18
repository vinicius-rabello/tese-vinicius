import torch
import numpy as np
from torch.utils.data import Dataset
import os


class SuperResDataset(Dataset):
    def __init__(self, hr_files, downsample_factor=4, freq_threshold=None):
        """
        Parameters
        ----------
        hr_files : list[str]
            List of yearly .npy files. Each file must have shape
            (N, 128, 128, 2).
        downsample_factor : int
            Downsampling factor used to generate LR inputs.
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