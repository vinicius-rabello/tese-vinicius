import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import torch.fft as fft

class SuperResDataset(Dataset):
    def __init__(self, hr_files, downsample_factor=4, freq_threshold=None):
        # Load HR data from .npy files
        self.hr_files = [np.load(file).astype(np.float32) for file in hr_files]
        print(self.hr_files[0].shape)
        self.hr_data = np.concatenate(self.hr_files, axis=0)[:,:,:,2:4]

        # Downscale and Upscale HR data to create LR data
        self.downsample_factor = downsample_factor
        self.freq_threshold = freq_threshold
        N, H, W, C = self.hr_data.shape
        f = self.downsample_factor
        if H % f != 0 or W % f != 0:
            raise ValueError(f"HR dimensions ({H}, {W}) must be divisible by downsample_factor ({f}).")
        
        reshaped_hr = self.hr_data.reshape(N, H//f, f, W//f, f, C)
        lr_data_downscaled = reshaped_hr.mean(axis=(2, 4))
        lr_data_upscaled_h = np.repeat(lr_data_downscaled, f, axis=1)
        self.lr_data_upscaled = np.repeat(lr_data_upscaled_h, f, axis=2)

    def apply_low_pass_filter(self, tensor, radius):
        C, H, W = tensor.shape
        
        freq_domain = torch.fft.fft2(tensor)
        freq_shifted = torch.fft.fftshift(freq_domain)

        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist = torch.sqrt((x - cx)**2 + (y - cy)**2)
        
        mask = dist < radius
        
        # 3. Aplica a máscara e reconstrói a imagem
        freq_shifted_filtered = freq_shifted * mask.to(freq_shifted.device)
        freq_filtered = torch.fft.ifftshift(freq_shifted_filtered)
        filtered_tensor = torch.fft.ifft2(freq_filtered)
        
        return filtered_tensor.real

    def __len__(self):
        return self.hr_data.shape[0]
    
    def __getitem__(self, idx):
        lr_img = torch.from_numpy(self.lr_data_upscaled[idx]).permute(2, 0, 1)
        hr_img = torch.from_numpy(self.hr_data[idx]).permute(2, 0, 1)

        # Aplicar filtro de frequência se o threshold for definido
        if self.freq_threshold is not None:
            lr_img = self.apply_low_pass_filter(lr_img, self.freq_threshold)

        return lr_img, hr_img

if __name__ == "__main__":
    dataset = SuperResDataset(hr_files=['data/100/window_2003.npy', 'data/100/window_2004.npy'], downsample_factor=1, freq_threshold=5)
    print(f"Dataset length: {len(dataset)}")

    # Get the first item (index 0)
    lr_sample, hr_sample = dataset[19]

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