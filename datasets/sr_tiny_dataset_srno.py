import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import math
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import torch.nn.functional as F

def resize_fn(img, size):
    # img: (H, W, C) numpy or tensor
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
    resized = F.interpolate(t, size=size, mode='bicubic', align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0)  # back to (H, W, C)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    #ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class SRTinyDataset(Dataset):
    def __init__(self, hr_files, downsample_factor=4, scale_min=1.0, scale_max=4.0, random_downsampling=True):
        # Load HR data from .npy files
        self.hr_files = [np.load(file).astype(np.float32) for file in hr_files]
        self.hr_data = np.concatenate(self.hr_files, axis=0)[:,:,:,2:4]

        # Downscale and Upscale HR data to create LR data
        self.downsample_factor = downsample_factor
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.random_downsampling = random_downsampling

        N, H, W, C = self.hr_data.shape
        f = self.downsample_factor
        if H % f != 0 or W % f != 0:
            raise ValueError(f"HR dimensions ({H}, {W}) must be divisible by downsample_factor ({f}) squared.")
        
        # First Downscale
        reshaped_hr = self.hr_data.reshape(N, H//f, f, W//f, f, C)
        self.hr_data = reshaped_hr.mean(axis=(2, 4))

    def __len__(self):
        return self.hr_data.shape[0]
    
    def __getitem__(self, idx):
        # Get HR image and convert to tensor
        img = self.hr_data[idx, :, :, :]
        if self.random_downsampling:
            s = random.uniform(self.scale_min, self.scale_max)
        else:
            s = self.downsample_factor
        
        h_lr = math.floor(img.shape[0] / s + 1e-9)
        w_lr = math.floor(img.shape[1] / s + 1e-9)
        # h_hr = round(h_lr * s)
        # w_hr = round(w_lr * s)
        # img = img[:, :h_hr, :w_hr]
        img_down = resize_fn(img, (h_lr, w_lr))
        crop_lr, crop_hr = img_down.permute(2, 0, 1), img.transpose((2, 0, 1))
        cell = torch.tensor([2 / crop_hr.shape[1], 2 / crop_hr.shape[2]], dtype=torch.float32)
        hr_coord = make_coord(img.shape[:2], flatten=False)
        hr_rgb = crop_hr
        # hr_tensor = torch.tensor(img, dtype=torch.float32)
        # hr_tensor = hr_tensor.permute(2, 0, 1)
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
    
if __name__ == "__main__":
    dataset = SRTinyDataset(hr_files=['data/100/window_2003.npy'], downsample_factor=4)
    sample = dataset[random.randint(0, len(dataset)-1)]
 
    # --- shapes and dtypes ---
    print('\nShapes:')
    for k, v in sample.items():
        print(f'  {k}: shape={tuple(v.shape)}, dtype={v.dtype}')
 
    # --- value ranges (check for NaNs, reasonable magnitudes) ---
    print('\nValue ranges:')
    print(f'  inp  min={sample["inp"].min():.4f}  max={sample["inp"].max():.4f}')
    print(f'  gt   min={sample["gt"].min():.4f}   max={sample["gt"].max():.4f}')
 
    # --- coord checks ---
    coord = sample['coord']
    print(f'\nCoord corners (should span [-1, 1]):')
    print(f'  top-left  (expect ~ [-1, -1]): {coord[0,  0].tolist()}')
    print(f'  top-right (expect ~ [-1, +1]): {coord[0, -1].tolist()}')
    print(f'  bot-left  (expect ~ [+1, -1]): {coord[-1, 0].tolist()}')
 
    # --- cell check ---
    hr_size = sample['gt'].shape[-1]
    expected_cell = round(2.0 / hr_size, 6)
    print(f'\nCell: {sample["cell"].tolist()}  (expected [{expected_cell}, {expected_cell}])')
 
    # --- save images for visual inspection ---
    plt.imsave('./lr_u.png', sample['inp'][0])
    plt.imsave('./lr_v.png', sample['inp'][1])
    plt.imsave('./hr_u.png', sample['gt'][0])
    plt.imsave('./hr_v.png', sample['gt'][1])
    print('\nSaved lr_u.png, lr_v.png, hr_u.png, hr_v.png')
    print('lr should look like a blocky low-res version of hr')