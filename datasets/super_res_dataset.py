import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2

class SuperResDataset(Dataset):
    def __init__(self, hr_files, downsample_factor=4):
        # Load HR data from .npy files
        self.hr_files = [np.load(file).astype(np.float32) for file in hr_files]
        print(self.hr_files[0].shape)
        self.hr_data = np.concatenate(self.hr_files, axis=0)[:,:,:,2:4]

        # Downscale and Upscale HR data to create LR data
        self.downsample_factor = downsample_factor
        N, H, W, C = self.hr_data.shape
        f = self.downsample_factor
        if H % f != 0 or W % f != 0:
            raise ValueError(f"HR dimensions ({H}, {W}) must be divisible by downsample_factor ({f}).")
        
        reshaped_hr = self.hr_data.reshape(N, H//f, f, W//f, f, C)
        lr_data_downscaled = reshaped_hr.mean(axis=(2, 4))
        lr_data_upscaled_h = np.repeat(lr_data_downscaled, f, axis=1)
        self.lr_data_upscaled = np.repeat(lr_data_upscaled_h, f, axis=2)

    def __len__(self):
        return self.hr_data.shape[0]
    
    def __getitem__(self, idx):
        # Get LR image and convert to tensor
        lr_img = self.lr_data_upscaled[idx, :, :, :]
        lr_tensor = torch.tensor(lr_img, dtype=torch.float32)
        lr_tensor = lr_tensor.permute(2, 0, 1)

        # Get HR image and convert to tensor
        hr_img = self.hr_data[idx, :, :, :]
        hr_tensor = torch.tensor(hr_img, dtype=torch.float32)
        hr_tensor = hr_tensor.permute(2, 0, 1)
        return lr_tensor, hr_tensor

if __name__ == "__main__":
    dataset = SuperResDataset(hr_files=['data/100/window_2003.npy', 'data/100/window_2004.npy'], downsample_factor=4)
    print(f"Dataset length: {len(dataset)}")

    # Get the first item (index 0)
    lr_sample, hr_sample = dataset[1987]

    # Print the shapes
    print(f"Shape of the returned Low-Resolution (LR) image: {lr_sample.shape}")
    print(f"Shape of the returned High-Resolution (HR) image: {hr_sample.shape}")
        
    # Optional: Check data types
    print(f"Dtype of LR image: {lr_sample.dtype}")
    print(f"Dtype of HR image: {hr_sample.dtype}")
    plt.imsave('lr_u.png', lr_sample.numpy()[0])
    plt.imsave('hr_u.png', hr_sample.numpy()[0])
    plt.imsave('lr_v.png', lr_sample.numpy()[1])
    plt.imsave('hr_v.png', hr_sample.numpy()[1])