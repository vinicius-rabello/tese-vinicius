from piq import ssim
import numpy as np
import cv2
from torchmetrics.image import PeakSignalNoiseRatio
import lpips
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

lpips_model = lpips.LPIPS(net='vgg').to(device)

def ssimLoss(iy,ihr,cmp,cnt):
    output = iy.cpu().squeeze().numpy()
    high = ihr.cpu().squeeze().numpy()

    lmin = min(output[cmp,:,:].min(),high[cmp,:,:].min()) * 1.01
    lmax = max(output[cmp,:,:].max(),high[cmp,:,:].max()) * 1.01
    if (lmin<0.0 and lmax<0.0):
        print("negative ",cnt)
    lmin = min(lmin,0.0)
    lmax = max(lmax,1.0)

    y = (iy.cpu()[:,cmp:cmp+1,:,:] - lmin) / (lmax - lmin)
    hr = (ihr.cpu()[:,cmp:cmp+1,:,:] - lmin) / (lmax - lmin)
    loss = ssim(y,hr,data_range=1.0)
    return loss.numpy()

psnr = PeakSignalNoiseRatio(data_range=1.0)

def psnrLoss(iy, ihr, cmp, cnt):
    output = iy.cpu().squeeze().numpy()  # Convert tensor to numpy
    high = ihr.cpu().squeeze().numpy()   # Convert tensor to numpy

    # Find the min and max values for scaling
    lmin = min(output[cmp,:,:].min(), high[cmp,:,:].min()) * 1.01
    lmax = max(output[cmp,:,:].max(), high[cmp,:,:].max()) * 1.01
    
    # Prevent negative values and enforce max value to be at least 1
    lmin = min(lmin, 0.0)
    lmax = max(lmax, 1.0)

    # Rescale the images to [0, 1]
    y = (iy.cpu()[:, cmp:cmp+1, :, :] - lmin) / (lmax - lmin)
    hr = (ihr.cpu()[:, cmp:cmp+1, :, :] - lmin) / (lmax - lmin)

    # Convert y and hr to numpy arrays before calculating MSE
    y = y.cpu().numpy()  # Convert tensor to numpy
    hr = hr.cpu().numpy()  # Convert tensor to numpy
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((y - hr) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')  # Infinite PSNR for identical images
    
    # Compute PSNR
    psnr = 10 * np.log10(1.0 / mse)  # Assuming pixel values are scaled to [0, 1]
    
    return psnr

def lpipsLoss(iy, ihr, cmp, cnt):
    """
    Compute LPIPS loss between input and high-resolution images.
    
    Args:
        iy (torch.Tensor): Low-resolution or generated image (B, C, H, W).
        ihr (torch.Tensor): High-resolution ground truth image (B, C, H, W).
        cmp (int): The channel to compare (e.g., 0 for first channel).
    
    Returns:
        float: LPIPS distance (lower is better).
    """
    # Extract the specified channel and normalize to [-1, 1]
    y = iy[:, cmp:cmp+1, :, :]
    hr = ihr[:, cmp:cmp+1, :, :]

    # Ensure input tensors are on GPU
    y = y.to(device)
    hr = hr.to(device)

    # Compute LPIPS loss
    loss = lpips_model(y, hr)

    return loss.mean().item()  # Convert to scalar value

def compute_fsim(y, hr):
    """
    Compute FSIM (Feature Similarity Index) between two images.

    Args:
        y (numpy.ndarray): Low-resolution output image.
        hr (numpy.ndarray): High-resolution ground truth image.

    Returns:
        float: FSIM score (higher means better quality).
    """

    # Convert images to grayscale if they are RGB
    if len(y.shape) == 3:
        y = cv2.cvtColor(y, cv2.COLOR_RGB2GRAY)
    if len(hr.shape) == 3:
        hr = cv2.cvtColor(hr, cv2.COLOR_RGB2GRAY)

    # Compute gradients using Sobel operator
    grad_y = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
    grad_hr = cv2.Sobel(hr, cv2.CV_64F, 1, 0, ksize=3)

    # Compute the phase congruency (an approximation)
    pc_y = np.abs(grad_y)
    pc_hr = np.abs(grad_hr)

    # Normalize both the gradient magnitudes
    norm_y = np.linalg.norm(pc_y)
    norm_hr = np.linalg.norm(pc_hr)

    # FSIM calculation
    fsim_score = np.sum(np.minimum(pc_y, pc_hr)) / np.sum(np.maximum(pc_y, pc_hr))
    
    return fsim_score
#-------------------------------------------------------------------------
def fsimLoss(iy, ihr, cmp, cnt):
    """
    Compute FSIM (Feature Similarity Index) between the generated (iy) and high-resolution (ihr) images.

    Args:
        iy (torch.Tensor): Generated image (low-resolution output from model).
        ihr (torch.Tensor): High-resolution ground truth image.
        cmp (int): Channel index (0 for first channel, 1 for second channel, etc.).

    Returns:
        float: FSIM score (higher means better quality).
    """
    y = iy.cpu().squeeze().numpy()
    hr = ihr.cpu().squeeze().numpy()

    y_channel = y[cmp, :, :]
    hr_channel = hr[cmp, :, :]

    fsim_score = compute_fsim(y_channel, hr_channel)

    return fsim_score

def epiLoss(iy, ihr, cmp, cnt):
    """
    Computes the Edge Preservation Index (EPI) between a low-resolution image and high-resolution ground truth.

    Parameters:
    iy (Tensor): The low-resolution image (PyTorch tensor).
    ihr (Tensor): The high-resolution image (PyTorch tensor).
    cmp (int): Channel index to compare (for multi-channel images).
    cnt (int): The index for the current comparison (useful for printing or tracking).

    Returns:
    float: The computed Edge Preservation Index.
    """
    # Convert to numpy arrays
    low_res = iy.cpu().squeeze().numpy()  # Convert low-resolution tensor to numpy array
    high_res = ihr.cpu().squeeze().numpy()  # Convert high-resolution tensor to numpy array

    # Extract the channel of interest
    low_res_channel = low_res[cmp, :, :]
    high_res_channel = high_res[cmp, :, :]

    # Compute gradients using Sobel operator
    grad_low_x = cv2.Sobel(low_res_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_low_y = cv2.Sobel(low_res_channel, cv2.CV_64F, 0, 1, ksize=3)
    grad_high_x = cv2.Sobel(high_res_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_high_y = cv2.Sobel(high_res_channel, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitudes
    grad_low = np.sqrt(grad_low_x**2 + grad_low_y**2)
    grad_high = np.sqrt(grad_high_x**2 + grad_high_y**2)

    # Normalize the gradients
    grad_low /= np.max(grad_low)
    grad_high /= np.max(grad_high)

    # Compute the Edge Preservation Index (EPI)
    epi = np.mean(grad_low * grad_high)

    return epi