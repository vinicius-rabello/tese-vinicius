import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from DSCMS import DSCMS
from loader import SuperResNpyDataset2
from PRUSR import PRUSR
import torch
import torch.nn.functional as F
import sys
from scipy.sparse.linalg import gmres
from piq import ssim
import cv2
from torchmetrics.image import PeakSignalNoiseRatio
import lpips



from DSCMS import DSCMS
from DSCMS_NN import DSCMS_NN
from DSCMS_PSS import DSCMS_PSS
from PRUSR import PRUSR
from CNN import CNN


###############################################################################################
def recoverVelocity(omega,lat,lon):
    # Define Earth radius (in meters)
    R = 6.371e6

    # Create 2D meshgrids for latitude and longitude
    lon_2d, lat_2d = np.deg2rad(lon),np.deg2rad(lat)

    # Calculate the grid spacing (assuming lat and lon are 2D grids)
    hy = R * np.gradient(lat_2d, axis=0)
    hx = R * np.cos(lat_2d) * np.gradient(lon_2d, axis=1)

    # Grid size
    Nx, Ny = lon.shape[1], lon.shape[0]  

    # Initialize streamfunction
    psi = np.zeros((Nx, Ny))

    # SOR Parameters
    omega_SOR = 0.1 #1.8  # Relaxation parameter
    tol = 1e-6
    max_iter = 20000

    #--------------------------------------------
    A = np.zeros((Nx*Ny,Nx*Ny))
    b = np.zeros(Nx*Ny)
    for j in range(0, Ny):
        for i in range(0, Nx):
            n = i + j * Nx

            hx_ip = hx[i, j] if i < Nx-1 else hx[i-1, j]
            hx_im = hx[i-1, j] if i > 0 else hx[i, j]
            hy_jp = hy[i, j] if j < Ny-1 else hy[i, j-1]
            hy_jm = hy[i, j-1] if j > 0 else hy[i, j]            

            A[n,n] =-2.0/hx_ip/hx_im - 2.0 / hy_jp / hy_jm
            #--
            if i>0:
                A[n,n-1] = 1.0/hx_im/hx_ip
            else:
                A[n,n] = A[n,n] #+ 1.0
            #--
            if i<Nx-1:
                A[n,n+1] = 1.0/hx_ip/hx_im
            else:
                A[n,n] = A[n,n] #+ 1.0
            #--
            if j>0:
                A[n,n-Nx] = 1.0/hy_jm/hy_jp
            else:
                A[n,n] = A[n,n] #+ 1.0
            #--
            if j<Ny-1:
                A[n,n+Nx] = 1.0/hy_jp/hy_jm
            else:
                A[n,n] = A[n,n] #+ 1.0
            #--
            b[n] =-omega[j,i]
        
    x = np.linalg.solve(A,b)
    #x, info = gmres(A, b, tol=1e-10, maxiter=1000)

    residual = b - A @ x
    residual_norm = np.linalg.norm(residual,ord=np.inf)
    
    psi = x.reshape((Nx,Ny))
    print("residual",psi.max(),omega.max(),residual_norm,max_iter)
    max_iter = -1
    #--------------------------------------------

    # Solve Poisson equation with non-uniform grid using SOR
    for it in range(max_iter):
        psi_old = psi.copy()

        for j in range(0, Ny):
            for i in range(0, Nx):
                hx_ip = hx[i, j] if i < Nx-1 else hx[i-1, j]
                hx_im = hx[i-1, j] if i > 0 else hx[i, j]
                hy_jp = hy[i, j] if j < Ny-1 else hy[i, j-1]
                hy_jm = hy[i, j-1] if j > 0 else hy[i, j]

                #hx_ip,hx_im,hy_jp,hy_jm = 1000.0,1000.0,1000.0,1000.0

                # Finite difference with non-uniform spacing
                A = 2 / (hx_ip * hx_im) + 2 / (hy_jp * hy_jm)

                # psi[i, j] = (1 - omega_SOR) * psi_old[i, j] + (omega_SOR / A) * (
                #     (psi_old[i+1, j] / (hx_ip * (hx_ip + hx_im)) if i < Nx-1 else psi_old[Nx-1, j] / (hx_ip * (hx_ip + hx_im))) +
                #     (psi_old[i-1, j] / (hx_im * (hx_ip + hx_im)) if i > 0 else psi_old[0, j] / (hx_im * (hx_ip + hx_im))) +
                #     (psi_old[i, j+1] / (hy_jp * (hy_jp + hy_jm)) if j < Ny-1 else psi_old[i, Ny-1] / (hy_jp * (hy_jp + hy_jm))) +
                #     (psi_old[i, j-1] / (hy_jm * (hy_jp + hy_jm)) if j > 0 else psi_old[i, 0] / (hy_jm * (hy_jp + hy_jm))) +
                #     omega[i, j]
                # )

                psi[i, j] = (1 - omega_SOR) * psi_old[i, j] + (omega_SOR / A) * (
                    (psi_old[i+1, j] / (hx_ip * hx_im) if i < Nx-1 else 0) +
                    (psi_old[i-1, j] / (hx_im * hx_ip) if i > 0 else 0) +
                    (psi_old[i, j+1] / (hy_jp * hy_jm) if j < Ny-1 else 0) +
                    (psi_old[i, j-1] / (hy_jm * hy_jp) if j > 0 else 0) +
                    omega[i, j]
                )



        # Check convergence
        error = np.linalg.norm(psi - psi_old, ord=np.inf)
        if it%1==0:
            print(it,error)
        if error < tol:
            print(f"Converged in {it} iterations")
            break

    # Compute velocity field (central differences on non-uniform grid)
    u = np.zeros((Nx, Ny))
    v = np.zeros((Nx, Ny))

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            hy_jp = hy[i, j]
            hy_jm = hy[i, j-1]
            hx_ip = hx[i, j]
            hx_im = hx[i-1, j]

            u[i, j] = (psi[i, j+1] - psi[i, j-1]) / (hy_jp + hy_jm)  # u = dψ/dy
            v[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (hx_ip + hx_im)  # v = -dψ/dx
    
    print(">>>",u.max(),u.min(),v.max(),v.min())
    return u,v

###############################################################################################
def Vorticity(u25,v25,lat,lon):

    # Define Earth radius (in meters)
    R = 6.371e6

    # Create 2D meshgrids for latitude and longitude
    lon_2d, lat_2d = np.deg2rad(lon),np.deg2rad(lat)

    # Calculate the grid spacing (assuming lat and lon are 2D grids)
    dy = R * np.gradient(lat_2d, axis=0)
    dx = R * np.cos(lat_2d) * np.gradient(lon_2d, axis=1)

    # Calculate partial derivatives using finite differences
    dvdx = np.gradient(v25, axis=1) / dx
    dudy = np.gradient(u25, axis=0) / dy

    # Calculate relative vorticity
    vorticity = dvdx - dudy

    return vorticity
################################################################################
def returnVal(X,meanX,stdX):
    return X * (stdX+1e-8) + meanX

# SSIM ###############################################################################
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

# PSNR ###############################################################################

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
##################################################################################


# Initialize the LPIPS loss model
lpips_model = lpips.LPIPS(net='vgg').cuda()  # Use 'alex' or 'squeeze' for alternative networks

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
    y = y.cuda()
    hr = hr.cuda()

    # Compute LPIPS loss
    loss = lpips_model(y, hr)

    return loss.mean().item()  # Convert to scalar value

# - FSIM - ##############################################################################
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
################################################################################
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


#    TKE           ###############################################################################
import torch
import torch.nn.functional as F

def LossTKE(hr, lr, channel, idx):
    """
    Compute Turbulent Kinetic Energy (TKE) loss between two images.

    Args:
        hr (torch.Tensor): High-resolution ground truth image [batch, channels, H, W].
        lr (torch.Tensor): Low-resolution predicted image [batch, channels, H, W].
        channel (int): Which channel to use (0 for u, 1 for v).
        idx (int): Index for tracking loop iteration (for debugging/logging purposes).

    Returns:
        float: TKE loss value.
    """
    # Ensure both tensors are on the same device
    device = hr.device  
    lr = lr.to(device)

    # Select the specified channel (0 for u, 1 for v)
    hr_channel = hr[:, channel, :, :]
    lr_channel = lr[:, channel, :, :]

    # Compute the difference (residuals)
    diff = lr_channel - hr_channel  # [batch, H, W]

    # Sobel filters for computing gradients
    sobel_filter_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32, device=device)
    sobel_filter_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32, device=device)

    # Reshape for convolution (batch, 1, H, W)
    diff = diff.unsqueeze(1)

    # Compute gradient differences
    grad_diff_x = F.conv2d(diff, sobel_filter_x, padding=1)
    grad_diff_y = F.conv2d(diff, sobel_filter_y, padding=1)

    # Compute TKE (0.5 * (u'^2 + v'^2))
    tke = 0.5 * (grad_diff_x**2 + grad_diff_y**2)

    # Return mean TKE over spatial dimensions
    return tke.mean().item()

# EK avaraged   ##############################################################################

from scipy.fft import fft2

def compute_energy_spectrum(velocity_field):
    """
    Compute the kinetic energy spectrum for a single snapshot of a 2-channel velocity field.

    Parameters:
    velocity_field (numpy.ndarray): Array of shape (2, Nx, Ny) where:
                                    velocity_field[0] -> u-component
                                    velocity_field[1] -> v-component

    Returns:
    tuple: (k_bins, E_k_radial) where:
           - k_bins: Wavenumber bins
           - E_k_radial: Radially averaged kinetic energy spectrum
    """
    u = velocity_field[0]
    v = velocity_field[1]
    
    # Compute mean velocity field
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    
    # Compute fluctuations
    u_prime = u - u_mean
    v_prime = v - v_mean
    
    # Grid size
    Nx, Ny = u.shape
    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k = np.sqrt(KX**2 + KY**2)  # Magnitude of wavenumber
    # np.set_printoptions(threshold=np.inf)  # Print all elements



    
    # Compute energy spectrum
    E_u = np.abs(fft2(u_prime))**2
    E_v = np.abs(fft2(v_prime))**2
    E_k = 0.5 * (E_u + E_v)
    
    # Radial binning


    some_value = 5  # Minimum number of points required for a valid bin
    valid_bins = []  # Store only reliable bins
    valid_energies = []  # Store only reliable energy values
    k_bins = np.arange(0.1, np.max(k)/2+1, 1)


    for i, k_min in enumerate(k_bins[:-1]):
        k_max = k_bins[i + 1]
        mask = (k >= k_min) & (k < k_max)
        count = np.count_nonzero(mask)
        # print(count)

        if count >= some_value:  # Only store valid bins
            valid_bins.append(k_min)
            valid_energies.append(np.mean(E_k[mask]))
        else:

            print(f"Skipping bin {i} (k = {k_min}) due to low count: {count}")

    # Convert lists back to arrays
    k_bins = np.array(valid_bins)
    E_k_radial = np.array(valid_energies)

    
    return k_bins, E_k_radial

###  EK separatly  ####################################################################################


def compute_energy_spectrum_separatly(u, v):
    """
    Compute the kinetic energy spectrum for a single snapshot of a 2-channel velocity field.

    Parameters:
    u (numpy.ndarray): u-component of velocity field (Nx, Ny)
    v (numpy.ndarray): v-component of velocity field (Nx, Ny)

    Returns:
    tuple: (k_bins, E_u_radial, E_v_radial) where:
           - k_bins: Wavenumber bins
           - E_u_radial: Radially averaged kinetic energy spectrum for U-component
           - E_v_radial: Radially averaged kinetic energy spectrum for V-component
    """
    # Compute mean velocity field
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    
    # Compute fluctuations
    u_prime = u - u_mean
    v_prime = v - v_mean
    
    # Grid size
    Nx, Ny = u.shape
    # kx = np.fft.fftfreq(Nx) * Nx
    # ky = np.fft.fftfreq(Ny) * Ny
    dx=dy=1000
    kx = np.fft.fftfreq(Nx, d=dx) * (2 * np.pi)  # Convert to physical wavenumber
    ky = np.fft.fftfreq(Ny, d=dy) * (2 * np.pi)


    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k = np.sqrt(KX**2 + KY**2)  # Magnitude of wavenumber
    
    # Compute energy spectrum for each component
    # E_u = np.abs(fft2(u_prime))**2
    # E_v = np.abs(fft2(v_prime))**2
    E_u = np.abs(fft2(u_prime))**2 / (Nx * Ny) 
    E_v = np.abs(fft2(v_prime))**2 / (Nx * Ny)  


    # Radial binning
    some_value = 5  # Minimum number of points required for a valid bin
    valid_bins = []  # Store only reliable bins
    valid_energies_u = []  # Store only reliable energy values for U
    valid_energies_v = []  # Store only reliable energy values for V
    # k_bins = np.arange(0.1, np.max(k)/2+1, 1)
    k_bins = np.logspace(np.log10(np.min(k[k>0])), np.log10(np.max(k)/2+1), num=64)




    for i, k_min in enumerate(k_bins[:-1]):
        k_max = k_bins[i + 1]
        mask = (k >= k_min) & (k < k_max)
        count = np.count_nonzero(mask)

        if count >= some_value:  # Only store valid bins
            valid_bins.append(k_min)
            valid_energies_u.append(np.mean(E_u[mask]))
            valid_energies_v.append(np.mean(E_v[mask]))
        else:
            print(f"Skipping bin {i} (k = {k_min}) due to low count: {count}")

    # Convert lists back to arrays
    k_bins = np.array(valid_bins)
    E_u_radial = np.array(valid_energies_u)
    E_v_radial = np.array(valid_energies_v)

    return k_bins, E_u_radial, E_v_radial



###############################################################################
################################################################################
year = int(sys.argv[1])
day = int(sys.argv[2])
#mid  = np.load("train/S50_M1_M24_I1D.dat.npy")
#highNP = np.load("../data/100/window_2003.npy")
lowNP = np.load(f"../data/100/window_{year:04d}.npy")
lat = lowNP[day,:,:,0]
lon = lowNP[day,:,:,1]
lowNP_u = lowNP[day,:,:,2]
lowNP_v = lowNP[day,:,:,3]
lowNP_o = lowNP[day,:,:,4]/1.0e-5
################################################################################
device = 'cuda'


#   MODELS     ##########################################################





model = DSCMS(2,2,3)
# model = CNN(2,2,3)




# model = torch.nn.DataParallel(model)
# model = model.to(device)



# model = PRUSR(2,2,1)
# model = PRUSR(2,2,3)    # Paper artichecture

model = torch.nn.DataParallel(model)
model = model.to(device)

# model = PRUSR(2,2).to(device)
# model = PRUSR(in_channels=2, num_layers=2).to(device).to(torch.float16)  # Mixed precision

################################################################################
### PSS   #####################
# model = CNN(in_channels=2, out_channels=2, factor_filter_num=2, upscale_factor=4).cuda()
# model = torch.nn.DataParallel(model)
################################################################################
# fixed window
#mean = np.array([-0.08672285 ,0.0166599])
#std = np.array([0.18411921,0.1535898])

# random window
mean = np.array([-0.00561308,0.07556629])
std = np.array([0.32576539,0.38299691])

################################################################################
# model.load_state_dict(torch.load('../weights/DSCMS_2i2o_HYCOM_2003_2006.pth'))
#model.load_state_dict(torch.load('../weights/DSCMS_2i2o_3L_HYCOM_Rand_2003_2006.pth'))

# model.load_state_dict(torch.load('../weights/MSE_train/MSE-order2/Reza_130.pth'))    # DSCMS LH
# model.load_state_dict(torch.load('../weights/train-DSCMS/Reza_110.pth'))  #DSCMS-HLH

# model.load_state_dict(torch.load('../weights/2D_Model1/Reza_90.pth'))    #  2D_MOdel1 (90 with 4x is the best results)
# model.load_state_dict(torch.load('../weights/2D_L1/Reza_145.pth'))    #  2D_MOdel1 (145 best results for L1)

model.load_state_dict(torch.load('../weights/2D_check1/Reza_145.pth'))    #  2D_MOdel1 (even numbers)



# model.load_state_dict(torch.load('../weights/train-CNN-HLH/Reza_130.pth'))    # CNN-HLH
# model.load_state_dict(torch.load('../weights/train-CNN/Reza_55.pth'))    # CNN 

# model.load_state_dict(torch.load('../weights/DSCMS-H0/Reza_50.pth'))    # DSCMS-H0
# model.load_state_dict(torch.load('../weights/CNN-H0/Reza_30.pth'))    # DSCMS-H0


# model.load_state_dict(torch.load('../weights/PRUSR_HLH/Reza_75.pth'))    # PRUSR -Original
# model.load_state_dict(torch.load('../weights/PRUSR_new/Reza_12.pth'))    # PRUSR -new




# from torchsummary import summary

# summary(model, input_size=(2, 128, 128))

# exit()


################################################################################"
data_folder = "../data"
lr_files = [f"25/window_{year:04d}.npy"]
hr_files = [f"100/window_{year:04d}.npy"]

dataset = SuperResNpyDataset2(data_folder, lr_files, hr_files,0,mean,std)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
################################################################################
vmin = 0.0 #e-5
vmax = 1.0 #e-5

trim = 4
#poch [96/2500], Loss Train.: 0.003492, Loss Val.: 0.003308, Learning rate: 1.25e-05
lowMin,lowMax   =  -11.453426448716854,23.47178113961668
highMin,highMax =  -27.36551687121391,45.74970516841858
################################################################################
bilinear_MSE_u,bilinear_MSE_v,bilinear_MSE_o = [],[],[]
bic_MSE_u,bic_MSE_v,bic_MSE_o = [],[],[]
bic_SSIM_u,bic_SSIM_v = [],[]
bic_PSNR_u,bic_PSNR_v = [],[]
model_PSNR_u,model_PSNR_v = [],[]
LR_lpips_u,LR_lpips_v = [],[]
model_lpips_u,model_lpips_v = [],[]
bilinear_lpips_u,bilinear_lpips_v =  [] , []
bic_lpips_u,bic_lpips_v =  [] , []
LR_FSIM_u,LR_FSIM_v = [],[]
model_FSIM_u,model_FSIM_v = [],[]
bilinear_FSIM_u,bilinear_FSIM_v =  [] , []
bic_FSIM_u,bic_FSIM_v =  [] , []
LR_EPI_u,LR_EPI_v = [],[]
model_EPI_u,model_EPI_v = [],[]
bilinear_EPI_u,bilinear_EPI_v =  [] , []
bic_EPI_u,bic_EPI_v =  [] , []
LR_FID_u,LR_FID_v = [],[]
LR_MSE_u,LR_MSE_v,LR_MSE_o = [],[],[]
LR_SSIM_u,LR_SSIM_v = [],[]
LR_PSNR_u,LR_PSNR_v = [],[]
bilinear_PSNR_u,bilinear_PSNR_v =  [] , []
model_MSE_u,model_MSE_v,model_MSE_o = [],[],[]
reverse_MSE_u,reverse_MSE_v,reverse_MSE_o =[],[],[]
model_SSIM_u,model_SSIM_v,bilinear_SSIM_u,bilinear_SSIM_v = [],[],[],[]
LR_TKE_u,LR_TKE_v = [],[]
model_TKE_u,model_TKE_v = [],[]
bilinear_TKE_u,bilinear_TKE_v =  [] , []
bic_TKE_u,bic_TKE_v =  [] , []
#####################################################################################




import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom

with torch.no_grad():
    for i, (lr, hr) in enumerate(test_loader):
        print(i)

        # print(np.shape(lr))
        lr = lr.to(device)  # Move LR to device
        hr = hr.to(device)  # Move HR to device
        # print(lr.min(),lr.max(),hr.min(),hr.max())
        loww = lr.cpu().squeeze().numpy()  # lr as a NumPy array (used for debugging)

        
        # print(np.shape(lr),np.shape(hr),"!!!!")
        # exit()
        # move to CPU and convert to numpy
        lr_np = lr.cpu().numpy()
        # lr_np = zoom(lr_np, zoom=(1, 1, 4, 4), order=3)
        lr = torch.tensor(lr_np).to(device)
        lr_upsampled = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)

        # lr = lr.to(device)  # Move LR to device
        # print(np.shape(lr),np.shape(hr))
        # exit()


        # Get the model output
        output = model(lr_upsampled)
        # output = model(lr)
        # output = torch.clamp(output, min=0.0, max=1.0)
        y = output  # Model's output (not used directly for SSIM)
        
        # lr_resized =  F.interpolate(lr, size=(128, 128), mode='bilinear', align_corners=False)  # mode='box' mode='nearest')
        # lr=lr_resized

        # print(np.shape(lr),np.shape(y),np.shape(lr_resized),np.shape(output))
        print(lr.min(),lr.max(),hr.min(),hr.max(), y.min(),y.max())
        # exit()

        # Convert to numpy for debugging and visualization (don't need to convert back to tensor)
        low = lr_upsampled.cpu().squeeze().numpy()  # lr as a NumPy array (used for debugging)
        high = hr.cpu().squeeze().numpy()  # hr as a NumPy array (used for debugging)
        output = output.cpu().squeeze().numpy()  # hr as a NumPy array (used for debugging)

        # print(np.shape(lr),np.shape(y),np.shape(lr_resized),np.shape(output))

        #-----------------------------------------------------------------------------------------
        #  Try to plot 
        # Example usage
        # velocity_field = np.random.randn(2, 128, 128)  # Random example data
        # velocity_field = output

        # k_bins, E_k_radial = compute_energy_spectrum(velocity_field)

        # # Plotting
        # plt.figure(figsize=(6, 4))
        # plt.loglog(k_bins, E_k_radial, label="Computed Spectrum")
        # plt.xlabel(r"$k$")
        # plt.ylabel(r"$E(k)$")
        # plt.title("Kinetic Energy Spectrum")
        # plt.legend()
        # plt.grid(True)
        # plt.savefig("kinetic_energy_spectrum.png", dpi=300, bbox_inches="tight")

        # # Optionally, close the figure to free memory
        # plt.close()

        # print("heer")
        # exit()

        #####################################################################################################
        #  -----------------  TKE FOR LR, SR, GT ===> u and v averaged   ----------------------------------------------
        #####################################################################################################
        # Assume output_low, output_medium, output_high are the three velocity fields
        # velocity_field_low = low
        # velocity_field_medium = output
        # velocity_field_high = high

        # # Compute energy spectra
        # k_bins_low, E_k_radial_low = compute_energy_spectrum(velocity_field_low)
        # k_bins_medium, E_k_radial_medium = compute_energy_spectrum(velocity_field_medium)
        # k_bins_high, E_k_radial_high = compute_energy_spectrum(velocity_field_high)

        # # Plotting
        # plt.figure(figsize=(6, 4))
        # plt.loglog(k_bins_low, E_k_radial_low, label="LR", linestyle="--", color="blue")
        # plt.loglog(k_bins_medium, E_k_radial_medium, label="SR", linestyle="-.", color="green")
        # plt.loglog(k_bins_high, E_k_radial_high, label="GT", linestyle="-", color="red")

        # plt.xlabel(r"$k$")
        # plt.ylabel(r"$E(k)$")
        # plt.title("Kinetic Energy Spectrum Comparison")
        # plt.legend()
        # plt.grid(True)

        # # Save the figure
        # plt.savefig("kinetic_LH.png", dpi=300, bbox_inches="tight")

        # # Optionally, close the figure to free memory
        # plt.close()

        # print("Comparison plot saved!")
        # exit()

        #####################################################################################################
        #  -----------------  TKE FOR LR, SR, GT ===> u and v Separatly   ----------------------------------------------
        #####################################################################################################



        # # Extract U and V components
        # low_U, low_V = low[0], low[1]
        # output_U, output_V = output[0], output[1]
        # high_U, high_V = high[0], high[1]

        #         # Compute energy spectra for U and V separately
        # k_bins_U_low, E_k_radial_U_low, E_k_radial_V_low = compute_energy_spectrum_separatly(low_U, low_V)
        # k_bins_U_medium, E_k_radial_U_medium, E_k_radial_V_medium = compute_energy_spectrum_separatly(output_U, output_V)
        # k_bins_U_high, E_k_radial_U_high, E_k_radial_V_high = compute_energy_spectrum_separatly(high_U, high_V)

        # # Now, let's plot the results for both U and V components separately:
        # fit = np.polyfit(np.log(k_bins_U_low), np.log(E_k_radial_U_low), 1)
        # print("Slope:", fit[0])


        # # Create subplots: one for U, one for V
        # fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # # Plot U component (Energy for U)
        # axes[0].loglog(k_bins_U_low, E_k_radial_U_low, label="LR U 1/25", linestyle="--", color="blue")
        # axes[0].loglog(k_bins_U_medium, E_k_radial_U_medium, label="SR U", linestyle="-.", color="green")
        # axes[0].loglog(k_bins_U_high, E_k_radial_U_high, label="GT U 1/100", linestyle="-", color="red")
        # # axes[0].loglog(k_bins_U_high, k_bins_U_high**(-5/3) * np.max(E_k_radial_U_high), '--', color="black", label="Kolmogorov -5/3")  # Kolmogorov slope

        # axes[0].set_xlabel(r"$k$")
        # axes[0].set_ylabel(r"$E_u(k)$")
        # axes[0].set_title("Kinetic Energy Spectrum (U component)")
        # axes[0].legend()
        # axes[0].grid(True)

        # # Plot V component (Energy for V)
        # axes[1].loglog(k_bins_U_low, E_k_radial_V_low, label="LR V 1/25", linestyle="--", color="blue")
        # axes[1].loglog(k_bins_U_medium, E_k_radial_V_medium, label="SR V", linestyle="-.", color="green")
        # axes[1].loglog(k_bins_U_high, E_k_radial_V_high, label="GT V 1/100", linestyle="-", color="red")
        # # axes[1].loglog(k_bins_U_high, k_bins_U_high**(-5/3) * np.max(E_k_radial_V_high), '--', color="black", label="Kolmogorov -5/3")  # Kolmogorov slope

        # axes[1].set_xlabel(r"$k$")
        # axes[1].set_ylabel(r"$E_v(k)$")
        # axes[1].set_title("Kinetic Energy Spectrum (V component)")
        # axes[1].legend()
        # axes[1].grid(True)

        # # Save the figure
        # plt.tight_layout()
        # plt.savefig("kinetic_UV_separately.png", dpi=300, bbox_inches="tight")

        # # Close the figure to free memory
        # plt.close()

        # print("Comparison plot for U and V components saved!")

        #####################################################################################################

        #---------------------------------------------------------------------------------------
        # print(type(low), type(high))  # Check their types
        # lr=lr_resized
        # exit()
        # print(np.shape(lr), np.shape(hr), np.shape(y), np.shape(low), np.shape(high), np.shape(output), "===========================")

        # # Convert low (NumPy array) to PyTorch tensor for resizing
        # low_tensor = torch.from_numpy(low).float().unsqueeze(0)  # Add batch dimension (1, 2, 32, 32)

        # # Resize the tensor to match high-resolution size (128, 128)
        # low_resized = F.interpolate(low_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        # hr_resized = F.interpolate(hr, size=(128, 128), mode='bilinear', align_corners=False)

        # # Ensure the resized tensor has the same number of channels as hr
        # low_resized_tensor = low_resized.squeeze(0)    # Remove batch dimension (2, 128, 128)

        # # Make sure the number of channels in low_resized_tensor matches hr
        # low_resized_tensor1 = low_resized_tensor.unsqueeze(0).numpy()  # Add batch dimension (1, 2, 128, 128)

        # # Ensure `low_resized_tensor` and `hr` are the same type and size
        # low_resized_tensor2 = low_resized_tensor.to(hr.dtype)  # Ensure same dtype
        # hr_resized = hr_resized.to(low_resized_tensor.dtype)  # Ensure same dtype

        # # If lr and hr are not the same size, resize them as well
        # lr_resized = F.interpolate(lr, size=(128, 128), mode='bilinear', align_corners=False)
        # lr_resized = lr_resized.to(hr_resized.dtype)  # Ensure same dtype

        # # Now `low_resized_tensor`, `hr_resized`, `lr_resized`, and `hr_resized` have the same size and type
        # low = low_resized_tensor.squeeze(0) 

        # print(np.shape(high), np.shape(low_resized_tensor1), np.shape(lr_resized), np.shape(hr_resized), "-=-=-=-=--=-")








    





   

        # print(lr.min(),lr.max(),hr.min(),hr.max())
        # exit()

        # LR Losses   ######################################################################################
        LR_MSE_u.append(np.mean(np.square(high[0,:,:] - low[0,:,:])))
        LR_MSE_v.append(np.mean(np.square(high[1,:,:] - low[1,:,:])))
 
        LR_SSIM_u.append(ssimLoss(hr,lr_upsampled,0,i))
        LR_SSIM_v.append(ssimLoss(hr,lr_upsampled,1,i))

        LR_PSNR_u.append(psnrLoss(hr,lr_upsampled,0,i))
        LR_PSNR_v.append(psnrLoss(hr,lr_upsampled,1,i))

        LR_lpips_u.append(lpipsLoss(hr,lr_upsampled,0,i))
        LR_lpips_v.append(lpipsLoss(hr,lr_upsampled,1,i))

        LR_FSIM_u.append(fsimLoss(hr,lr_upsampled,0,i))
        LR_FSIM_v.append(fsimLoss(hr,lr_upsampled,0,i))

        LR_EPI_u.append(epiLoss(hr,lr_upsampled,0,i))
        LR_EPI_v.append(epiLoss(hr,lr_upsampled,1,i))
        LR_TKE_u.append(LossTKE(hr,lr_upsampled,0,i))
        LR_TKE_v.append(LossTKE(hr,lr_upsampled,1,i))


        # Model Losses ######################################################################################

        model_MSE_u.append(np.mean(np.square(high[0,:,:] - output[0,:,:])))
        model_MSE_v.append(np.mean(np.square(high[1,:,:] - output[1,:,:])))

        model_SSIM_u.append(ssimLoss(y,hr,0,i))
        model_SSIM_v.append(ssimLoss(y,hr,1,i))

        model_PSNR_u.append(psnrLoss(y,hr,0,i))
        model_PSNR_v.append(psnrLoss(y,hr,1,i))

        model_lpips_u.append(lpipsLoss(y,hr,0,i))
        model_lpips_v.append(lpipsLoss(y,hr,1,i))

        model_FSIM_u.append(fsimLoss(y,hr,0,i))
        model_FSIM_v.append(fsimLoss(y,hr,1,i))

        model_EPI_u.append(epiLoss(y,hr,0,i))
        model_EPI_v.append(epiLoss(y,hr,1,i))

        model_TKE_u.append(LossTKE(y,hr,0,i))
        model_TKE_v.append(LossTKE(y,hr,1,i))


        #low = low*(highMax-highMin)*1.01+highMin*1.01
        #high = high*(highMax-highMin)*1.01+highMin*1.01
        #output = output*(highMax-highMin)*1.01+highMin*1.01        
        
        # low = low[:,:].reshape(2,int(low.shape[1]/4), 4, int(low.shape[2]/4), 4)
        # low = low.mean(axis=(2, 4))

        # reshaped_hr = high[:,:].reshape(2,int(high.shape[1]/4), 4, int(high.shape[2]/4), 4)
        # low = reshaped_hr.mean(axis=(2, 4))
        # low = low.mean(axis=(2, 4))

        #pos = low[0:2,:,:]
        #low = low[2:4,:,:]

        # mean_low = low.mean()
        # std_low = low.std()

        # mean_hr = high.mean()
        # std_hr = high.std()


        # print("#########################################################################")
        # print("############    BEfore Normalization            ###########################")
        # print("#########################################################################")

        # print("--- lr & hr :  Mean - std  --------------------------------------------")
        # print("##########################################################################")

        # print("Mean (LR) :: Std (LR) = ", mean_low, std_low)
        # print("Mean (HR) :: Std (HR) = ", mean_hr, std_hr)

        # print("##########################################################################")
        # print("--- lr & hr : Min/Max      --------------------------------------------")
        # print("##########################################################################")

        # print("lr minMax = ",low.min(),low.max())
        # print("hr minMax = ",high.min(),high.max(),"////////////////////")
        # print("##########################################################################")
        # exit()

        ### bilinear ##############################################################################
        # x = torch.tensor(low.reshape(2,low.shape[1],low.shape[2]), dtype=torch.float32).to(device)
        x = torch.tensor(loww.reshape(2,loww.shape[1],loww.shape[2]), dtype=torch.float32).to(device)
        x8 = F.interpolate(x.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)
        # exit()

        bilinear_SSIM_u.append(ssimLoss(x8,hr,0,i))
        bilinear_SSIM_v.append(ssimLoss(x8,hr,1,i))

        bilinear_PSNR_u.append(psnrLoss(x8,hr,0,i))
        bilinear_PSNR_v.append(psnrLoss(x8,hr,1,i))

        bilinear_lpips_u.append(lpipsLoss(x8,hr,0,i))
        bilinear_lpips_v.append(lpipsLoss(x8,hr,1,i))

        bilinear_FSIM_u.append(fsimLoss(x8,hr,0,i))
        bilinear_FSIM_v.append(fsimLoss(x8,hr,1,i))

        bilinear_EPI_u.append(epiLoss(x8,hr,0,i))
        bilinear_EPI_v.append(epiLoss(x8,hr,1,i))

        bilinear_TKE_u.append(LossTKE(x8,hr,0,i))
        bilinear_TKE_v.append(LossTKE(x8,hr,1,i))


        x8 = x8.cpu().squeeze(0).numpy() 

        bilinear_MSE_u.append(np.mean(np.square(high[0,:,:] - x8[0,:,:])))
        bilinear_MSE_v.append(np.mean(np.square(high[1,:,:] - x8[1,:,:])))





        ###  bicubic     #############################################################################

        x8c = F.interpolate(x.unsqueeze(0), scale_factor=4, mode='bicubic', align_corners=False)

        bic_SSIM_u.append(ssimLoss(x8c,hr,0,i))
        bic_SSIM_v.append(ssimLoss(x8c,hr,1,i))   

        bic_PSNR_u.append(psnrLoss(x8c,hr,0,i))
        bic_PSNR_v.append(psnrLoss(x8c,hr,1,i))

        bic_lpips_u.append(lpipsLoss(x8c,hr,0,i))
        bic_lpips_v.append(lpipsLoss(x8c,hr,1,i))

        bic_FSIM_u.append(fsimLoss(x8c,hr,0,i))
        bic_FSIM_v.append(fsimLoss(x8c,hr,1,i))

        bic_EPI_u.append(epiLoss(x8c,hr,0,i))
        bic_EPI_v.append(epiLoss(x8c,hr,1,i))

        bic_TKE_u.append(LossTKE(x8c,hr,0,i))
        bic_TKE_v.append(LossTKE(x8c,hr,1,i))


        x8c = x8c.cpu().squeeze(0).numpy()

        bic_MSE_u.append(np.mean(np.square(high[0,:,:] - x8c[0,:,:])))
        bic_MSE_v.append(np.mean(np.square(high[1,:,:] - x8c[1,:,:])))

        ################################################################################
        # standardization function







        # low[0,:,:] = returnVal(low[0,:,:],mean[0],std[0])
        # low[1,:,:] = returnVal(low[1,:,:],mean[1],std[1])

        

        if (i==day):
            #----------------------------------------------------------
            vmin1,vmax1 = 0.9*low[0,:,:].min(),1.1*low[0,:,:].max()
            vmin2,vmax2 = 0.9*low[1,:,:].min(),1.1*low[1,:,:].max()
            #----------------------------------------------------------
            plt.figure(figsize=(20, 12))
            fig, axes = plt.subplots(2, 5, figsize=(20, 12))
            axes = axes.flatten()
            #----------------------------------------------------------
            fig.text(0.02, 0.70 - 0 * 0.25, "u", va='center', ha='left', fontsize=12, fontweight='bold', rotation=90)
            fig.text(0.02, 0.70 - 1 * 0.45, "v", va='center', ha='left', fontsize=12, fontweight='bold', rotation=90)
            #----------------------------------------------------------
            axes[0].set_title('Low Resolution')
            axes[0].imshow(low[0,:,:],vmin=vmin1, vmax=vmax1)
            # axes[0].set_xlabel(f"MSE {LR_MSE_u[-1]:0.06f} | SSIM {LR_SSIM_u[-1]:0.06f}", labelpad=8)
            axes[0].set_xlabel(
                f"$\\bf{{MSE}}$: {LR_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {LR_SSIM_u[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {LR_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {LR_lpips_u[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {LR_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {LR_EPI_u[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {LR_TKE_u[-1]:0.06f}",
                labelpad=8, fontsize=12
            )
            axes[1].set_title('2D_error')
            axes[1].imshow(x8[0,:,:],vmin=vmin1, vmax=vmax1)
            # axes[1].set_xlabel(f"MSE {bilinear_MSE_u[-1]:0.06f} | SSIM {bilinear_SSIM_u[-1]:0.06f}", labelpad=8)
            axes[1].set_xlabel(
                f"$\\bf{{MSE}}$: {bilinear_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {bilinear_SSIM_u[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {bilinear_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bilinear_lpips_u[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {bilinear_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {bilinear_EPI_u[-1]:0.06f}\n"
                 f"$\\bf{{TKE}}$: {bilinear_TKE_u[-1]:0.06f}",
                labelpad=8, fontsize=12
            )
            axes[2].set_title('Output')
            axes[2].imshow(x8c[0,:,:],vmin=vmin1, vmax=vmax1)
            # axes[2].set_xlabel(f"MSE {bic_MSE_u[-1]:0.06f} | SSIM {bic_SSIM_u[-1]:0.06f}", labelpad=8)
            axes[2].set_xlabel(
                f"$\\bf{{MSE}}$: {bic_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {bic_SSIM_u[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {bic_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bic_lpips_u[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {bic_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {bic_EPI_u[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {bic_TKE_u[-1]:0.06f}",
                labelpad=8, fontsize=12
            )

            axes[3].set_title('Out+2Derr')
            axes[3].imshow(output[0,:,:],vmin=vmin1, vmax=vmax1)
            # axes[3].set_xlabel(f"MSE {model_MSE_u[-1]:0.06f} | SSIM {model_SSIM_u[-1]:0.06f}", labelpad=8)
            axes[3].set_xlabel(
                f"$\\bf{{MSE}}$: {model_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {model_SSIM_u[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {model_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {model_lpips_u[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {model_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {model_EPI_u[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {model_TKE_u[-1]:0.06f}",
                labelpad=8, fontsize=12
            )

            axes[4].set_title('High Resolution')
            axes[4].imshow(high[0,:,:],vmin=vmin1, vmax=vmax1)

            axes[5].imshow(low[1,:,:],vmin=vmin2, vmax=vmax2)
            # axes[5].set_xlabel(f"MSE {LR_MSE_v[-1]:0.06f} | SSIM {LR_SSIM_v[-1]:0.06f}", labelpad=8)
            axes[5].set_xlabel(
                f"$\\bf{{MSE}}$: {LR_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {LR_SSIM_v[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {LR_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {LR_lpips_v[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {LR_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {LR_EPI_v[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {LR_TKE_v[-1]:0.06f}",
                labelpad=8, fontsize=12
            )

            axes[6].imshow(x8[1,:,:],vmin=vmin2, vmax=vmax2)
            # axes[6].set_xlabel(f"MSE {bilinear_MSE_v[-1]:0.06f} | SSIM {bilinear_SSIM_v[-1]:0.06f}", labelpad=8)
            axes[6].set_xlabel(
                f"$\\bf{{MSE}}$: {bilinear_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {bilinear_SSIM_v[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {bilinear_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bilinear_lpips_v[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {bilinear_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {bilinear_EPI_v[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {bilinear_TKE_v[-1]:0.06f}",

                labelpad=8, fontsize=12
            )


            axes[7].imshow(x8c[1,:,:],vmin=vmin2, vmax=vmax2)
            # axes[7].set_xlabel(f"MSE {bic_MSE_v[-1]:0.06f} | SSIM {bic_SSIM_v[-1]:0.06f}", labelpad=8)
            axes[7].set_xlabel(
                f"$\\bf{{MSE}}$: {bic_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {bic_SSIM_v[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {bic_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bic_lpips_v[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {bic_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {bic_EPI_v[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {bic_TKE_v[-1]:0.06f}",
                labelpad=8, fontsize=12
            )

            axes[8].imshow(output[1,:,:],vmin=vmin2, vmax=vmax2)
            # axes[8].set_xlabel(f"MSE {model_MSE_v[-1]:0.06f} | SSIM {model_SSIM_v[-1]:0.06f}", labelpad=8)
            axes[8].set_xlabel(
                f"$\\bf{{MSE}}$: {model_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {model_SSIM_v[-1]:0.06f}\n"
                f"$\\bf{{PSNR}}$: {model_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {model_lpips_v[-1]:0.06f}\n"
                f"$\\bf{{FSIM}}$: {model_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {model_EPI_v[-1]:0.06f}\n"
                f"$\\bf{{TKE}}$: {model_TKE_v[-1]:0.06f}",
                labelpad=8, fontsize=12
            )

            axes[9].imshow(high[1,:,:],vmin=vmin2, vmax=vmax2)       


            # fig.suptitle(f"DSCMS :: Trained on HYCOM (2003-2006) :: Downsample HR :: Model 3->3 [u,v,Omega] :: Tested on Year {year} Record {day}", fontsize=16)
            fig.suptitle(f"MinMax :: Trained on HYCOM (2003-2006) :: Downsample HR :: $\\bf{{Trained\\,by\\,MSE}}$ :: Tested on Year {year} Record {day}", fontsize=16)



            #plt.tight_layout()
            plt.subplots_adjust(left=0.05)
            plt.savefig("PLOT(check).png",dpi=300,bbox_inches='tight')
            # plt.savefig("PLOT_CNN_105.png",dpi=300,bbox_inches='tight')
            exit()
            #print(l2,l2x)



        # if (i==day):
        #     #----------------------------------------------------------
        #     vmin1,vmax1 = 0.9*low[0,:,:].min(),1.1*low[0,:,:].max()
        #     vmin2,vmax2 = 0.9*low[1,:,:].min(),1.1*low[1,:,:].max()
        #     #----------------------------------------------------------
        #     plt.figure(figsize=(20, 12))
        #     fig, axes = plt.subplots(2, 5, figsize=(20, 12))
        #     axes = axes.flatten()
        #     #----------------------------------------------------------
        #     fig.text(0.02, 0.70 - 0 * 0.25, "u", va='center', ha='left', fontsize=12, fontweight='bold', rotation=90)
        #     fig.text(0.02, 0.70 - 1 * 0.45, "v", va='center', ha='left', fontsize=12, fontweight='bold', rotation=90)
        #     #----------------------------------------------------------
        #     axes[0].set_title('Low Resolution')
        #     axes[0].imshow(low[0,:,:],vmin=vmin1, vmax=vmax1)
        #     # axes[0].set_xlabel(f"MSE {LR_MSE_u[-1]:0.06f} | SSIM {LR_SSIM_u[-1]:0.06f}", labelpad=8)
        #     axes[0].set_xlabel(
        #         f"$\\bf{{MSE}}$: {LR_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {LR_SSIM_u[-1]:0.06f}\n"
        #         f"$\\bf{{PSNR}}$: {LR_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {LR_lpips_u[-1]:0.06f}\n"
        #         f"$\\bf{{FSIM}}$: {LR_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {LR_EPI_u[-1]:0.06f}\n"
        #         f"$\\bf{{TKE}}$: {LR_TKE_u[-1]:0.06f}",
        #         labelpad=8, fontsize=12
        #     )
        #     axes[1].set_title('bilinear')
        #     axes[1].imshow(returnVal(x8[0,:,:],mean[0],std[0]),vmin=vmin1, vmax=vmax1)
        #     # axes[1].set_xlabel(f"MSE {bilinear_MSE_u[-1]:0.06f} | SSIM {bilinear_SSIM_u[-1]:0.06f}", labelpad=8)
        #     axes[1].set_xlabel(
        #         f"$\\bf{{MSE}}$: {bilinear_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {bilinear_SSIM_u[-1]:0.06f}\n"
        #         f"$\\bf{{PSNR}}$: {bilinear_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bilinear_lpips_u[-1]:0.06f}\n"
        #         f"$\\bf{{FSIM}}$: {bilinear_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {bilinear_EPI_u[-1]:0.06f}\n"
        #          f"$\\bf{{TKE}}$: {bilinear_TKE_u[-1]:0.06f}",
        #         labelpad=8, fontsize=12
        #     )
        #     axes[2].set_title('bic')
        #     axes[2].imshow(returnVal(x8c[0,:,:],mean[0],std[0]),vmin=vmin1, vmax=vmax1)
        #     # axes[2].set_xlabel(f"MSE {bic_MSE_u[-1]:0.06f} | SSIM {bic_SSIM_u[-1]:0.06f}", labelpad=8)
        #     axes[2].set_xlabel(
        #         f"$\\bf{{MSE}}$: {bic_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {bic_SSIM_u[-1]:0.06f}\n"
        #         f"$\\bf{{PSNR}}$: {bic_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bic_lpips_u[-1]:0.06f}\n"
        #         f"$\\bf{{FSIM}}$: {bic_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {bic_EPI_u[-1]:0.06f}\n"
        #         f"$\\bf{{TKE}}$: {bic_TKE_u[-1]:0.06f}",
        #         labelpad=8, fontsize=12
        #     )

        #     axes[3].set_title('Super-Resolved')
        #     axes[3].imshow(returnVal(output[0,:,:],mean[0],std[0]),vmin=vmin1, vmax=vmax1)
        #     # axes[3].set_xlabel(f"MSE {model_MSE_u[-1]:0.06f} | SSIM {model_SSIM_u[-1]:0.06f}", labelpad=8)
        #     axes[3].set_xlabel(
        #         f"$\\bf{{MSE}}$: {model_MSE_u[-1]:0.06f} \t $\\bf{{SSIM}}$: {model_SSIM_u[-1]:0.06f}\n"
        #         f"$\\bf{{PSNR}}$: {model_PSNR_u[-1]:0.06f} \t $\\bf{{LPIPS}}$: {model_lpips_u[-1]:0.06f}\n"
        #         f"$\\bf{{FSIM}}$: {model_FSIM_u[-1]:0.06f} \t $\\bf{{EPI}}$: {model_EPI_u[-1]:0.06f}\n"
        #         f"$\\bf{{TKE}}$: {model_TKE_u[-1]:0.06f}",
        #         labelpad=8, fontsize=12
        #     )

        #     axes[4].set_title('High Resolution')
        #     axes[4].imshow(returnVal(high[0,:,:],mean[0],std[0]),vmin=vmin1, vmax=vmax1)

        #     axes[5].imshow(low[1,:,:],vmin=vmin2, vmax=vmax2)
        #     # axes[5].set_xlabel(f"MSE {LR_MSE_v[-1]:0.06f} | SSIM {LR_SSIM_v[-1]:0.06f}", labelpad=8)
        #     axes[5].set_xlabel(
        #         f"$\\bf{{MSE}}$: {LR_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {LR_SSIM_v[-1]:0.06f}\n"
        #         f"$\\bf{{PSNR}}$: {LR_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {LR_lpips_v[-1]:0.06f}\n"
        #         f"$\\bf{{FSIM}}$: {LR_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {LR_EPI_v[-1]:0.06f}\n"
        #         f"$\\bf{{TKE}}$: {LR_TKE_v[-1]:0.06f}",
        #         labelpad=8, fontsize=12
        #     )

        #     axes[6].imshow(returnVal(x8[1,:,:],mean[1],std[1]),vmin=vmin2, vmax=vmax2)
        #     # axes[6].set_xlabel(f"MSE {bilinear_MSE_v[-1]:0.06f} | SSIM {bilinear_SSIM_v[-1]:0.06f}", labelpad=8)
        #     axes[6].set_xlabel(
        #         f"$\\bf{{MSE}}$: {bilinear_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {bilinear_SSIM_v[-1]:0.06f}\n"
        #         f"$\\bf{{PSNR}}$: {bilinear_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bilinear_lpips_v[-1]:0.06f}\n"
        #         f"$\\bf{{FSIM}}$: {bilinear_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {bilinear_EPI_v[-1]:0.06f}\n"
        #         f"$\\bf{{TKE}}$: {bilinear_TKE_v[-1]:0.06f}",

        #         labelpad=8, fontsize=12
        #     )


        #     axes[7].imshow(returnVal(x8[1,:,:],mean[1],std[1]),vmin=vmin2, vmax=vmax2)
        #     # axes[7].set_xlabel(f"MSE {bic_MSE_v[-1]:0.06f} | SSIM {bic_SSIM_v[-1]:0.06f}", labelpad=8)
        #     axes[7].set_xlabel(
        #         f"$\\bf{{MSE}}$: {bic_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {bic_SSIM_v[-1]:0.06f}\n"
        #         f"$\\bf{{PSNR}}$: {bic_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {bic_lpips_v[-1]:0.06f}\n"
        #         f"$\\bf{{FSIM}}$: {bic_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {bic_EPI_v[-1]:0.06f}\n"
        #         f"$\\bf{{TKE}}$: {bic_TKE_v[-1]:0.06f}",
        #         labelpad=8, fontsize=12
        #     )

        #     axes[8].imshow(returnVal(output[1,:,:],mean[1],std[1]),vmin=vmin2, vmax=vmax2)
        #     # axes[8].set_xlabel(f"MSE {model_MSE_v[-1]:0.06f} | SSIM {model_SSIM_v[-1]:0.06f}", labelpad=8)
        #     axes[8].set_xlabel(
        #         f"$\\bf{{MSE}}$: {model_MSE_v[-1]:0.06f} \t $\\bf{{SSIM}}$: {model_SSIM_v[-1]:0.06f}\n"
        #         f"$\\bf{{PSNR}}$: {model_PSNR_v[-1]:0.06f} \t $\\bf{{LPIPS}}$: {model_lpips_v[-1]:0.06f}\n"
        #         f"$\\bf{{FSIM}}$: {model_FSIM_v[-1]:0.06f} \t $\\bf{{EPI}}$: {model_EPI_v[-1]:0.06f}\n"
        #         f"$\\bf{{TKE}}$: {model_TKE_v[-1]:0.06f}",
        #         labelpad=8, fontsize=12
        #     )

        #     axes[9].imshow(returnVal(high[1,:,:],mean[1],std[1]),vmin=vmin2, vmax=vmax2)       


        #     # fig.suptitle(f"DSCMS :: Trained on HYCOM (2003-2006) :: Downsample HR :: Model 3->3 [u,v,Omega] :: Tested on Year {year} Record {day}", fontsize=16)
        #     fig.suptitle(f"PRUSR_HLH :: Trained on HYCOM (2003-2006) :: Downsample HR :: $\\bf{{Trained\\,by\\,MSE}}$ :: Tested on Year {year} Record {day}", fontsize=16)



        #     #plt.tight_layout()
        #     plt.subplots_adjust(left=0.05)
        #     plt.savefig("2D_MOdel1.png",dpi=300,bbox_inches='tight')
        #     exit()
        #     # plt.savefig("PLOT_CNN_105.png",dpi=300,bbox_inches='tight')
           
        #     #print(l2,l2x)


################################################################################

# LR array ###############################################################################
LR_MSE_u = np.array(LR_MSE_u)
LR_MSE_v = np.array(LR_MSE_v)

LR_SSIM_u = np.array(LR_SSIM_u)
LR_SSIM_v = np.array(LR_SSIM_v)

LR_PSNR_u = np.array(LR_PSNR_u)
LR_PSNR_v = np.array(LR_PSNR_v)

LR_lpips_u = np.array(LR_lpips_u)
LR_lpips_v = np.array(LR_lpips_v)

LR_FSIM_u = np.array(LR_FSIM_u)
LR_FSIM_v = np.array(LR_FSIM_v)

LR_EPI_u = np.array(LR_EPI_u)
LR_EPI_v = np.array(LR_EPI_v)

LR_TKE_u = np.array(LR_TKE_u)
LR_TKE_v = np.array(LR_TKE_v)


# bilinear array ###############################################################################
bilinear_MSE_u = np.array(bilinear_MSE_u)
bilinear_MSE_v = np.array(bilinear_MSE_v)

bilinear_SSIM_u = np.array(bilinear_SSIM_u)
bilinear_SSIM_v = np.array(bilinear_SSIM_v)

bilinear_PSNR_u = np.array(bilinear_PSNR_u)
bilinear_PSNR_v = np.array(bilinear_PSNR_v)

bilinear_lpips_u = np.array(bilinear_lpips_u)
bilinear_lpips_v = np.array(bilinear_lpips_v)

bilinear_FSIM_u = np.array(bilinear_FSIM_u)
bilinear_FSIM_v = np.array(bilinear_FSIM_v)

bilinear_EPI_u = np.array(bilinear_EPI_u)
bilinear_EPI_v = np.array(bilinear_EPI_v)

bilinear_TKE_u = np.array(bilinear_TKE_u)
bilinear_TKE_v = np.array(bilinear_TKE_v)
# bic array ###############################################################################

bic_MSE_u = np.array(bic_MSE_u)
bic_MSE_v = np.array(bic_MSE_v)

bic_SSIM_u = np.array(bic_SSIM_u)
bic_SSIM_v = np.array(bic_SSIM_v)

bic_PSNR_u = np.array(bic_PSNR_u)
bic_PSNR_v = np.array(bic_PSNR_v)

bic_lpips_u = np.array(bic_lpips_u)
bic_lpips_v = np.array(bic_lpips_v)

bic_FSIM_u = np.array(bic_FSIM_u)
bic_FSIM_v = np.array(bic_FSIM_v)

bic_EPI_u = np.array(bic_EPI_u)
bic_EPI_v = np.array(bic_EPI_v)

bic_TKE_u = np.array(bic_TKE_u)
bic_TKE_v = np.array(bic_TKE_v)

# model array ###############################################################################

model_MSE_u = np.array(model_MSE_u)
model_MSE_v = np.array(model_MSE_v)

model_SSIM_u = np.array(model_SSIM_u)
model_SSIM_v = np.array(model_SSIM_v)

model_PSNR_u = np.array(model_PSNR_u)
model_PSNR_v = np.array(model_PSNR_v)

model_lpips_u = np.array(model_lpips_u)
model_lpips_v = np.array(model_lpips_v)

model_FSIM_u = np.array(model_FSIM_u)
model_FSIM_v = np.array(model_FSIM_v)

model_EPI_u = np.array(model_EPI_u)
model_EPI_v = np.array(model_EPI_v)

model_TKE_u = np.array(model_TKE_u)
model_TKE_v = np.array(model_TKE_v)




print("# - MSE - #################################################")

print("MSE LR_u = ",LR_MSE_u.mean(),LR_MSE_u.std())
print("MSE LR_v = ",LR_MSE_v.mean(),LR_MSE_v.std())

print("MSE bilinear_u = ",bilinear_MSE_u.mean(),bilinear_MSE_u.std())
print("MSE bilinear_v = ",bilinear_MSE_v.mean(),bilinear_MSE_v.std())

print("MSE bic_u = ",bic_MSE_u.mean(),bic_MSE_u.std())
print("MSE bic_v = ",bic_MSE_v.mean(),bic_MSE_v.std())

print("MSE Model_u = ",model_MSE_u.mean(),model_MSE_u.std())
print("MSE Model_v = ",model_MSE_v.mean(),model_MSE_v.std())



print("# - SSIM -  #################################################")

print("SSIM LR_u = ",LR_SSIM_u.mean(),LR_SSIM_u.std())
print("SSIM LR_v = ",LR_SSIM_v.mean(),LR_SSIM_v.std())

print("SSIM bilinear_u = ",bilinear_SSIM_u.mean(),bilinear_SSIM_u.std())
print("SSIM bilinear_v = ",bilinear_SSIM_v.mean(),bilinear_SSIM_v.std())

print("SSIM bic_u = ",bic_SSIM_u.mean(),bic_SSIM_u.std())
print("SSIM bic_v = ",bic_SSIM_v.mean(),bic_SSIM_v.std())

print("SSIM Model_u = ",model_SSIM_u.mean(),model_SSIM_u.std())
print("SSIM Model_v = ",model_SSIM_v.mean(),model_SSIM_v.std())

print("# - PSNR - #################################################")
print("PSNR LR_u = ",LR_PSNR_u.mean(),LR_PSNR_u.std())
print("PSNR LR_v = ",LR_PSNR_v.mean(),LR_PSNR_v.std())

print("PSNR bilinear_u = ",bilinear_PSNR_u.mean(),bilinear_PSNR_u.std())
print("PSNR bilinear_v = ",bilinear_PSNR_v.mean(),bilinear_PSNR_v.std())

print("PSNR bic_u = ",bic_PSNR_u.mean(),bic_PSNR_u.std())
print("PSNR bic_v = ",bic_PSNR_v.mean(),bic_PSNR_v.std())

print("PSNR model_u = ",model_PSNR_u.mean(),model_PSNR_u.std())
print("PSNR model_v = ",model_PSNR_v.mean(),model_PSNR_v.std())



print("# - lpips -  #################################################")
print("lpips LR_u = ",LR_lpips_u.mean(),LR_lpips_u.std())
print("lpips LR_v = ",LR_lpips_v.mean(),LR_lpips_v.std())

print("lpips bilinear_u = ",bilinear_lpips_u.mean(),bilinear_lpips_u.std())
print("lpips bilinear_v = ",bilinear_lpips_v.mean(),bilinear_lpips_v.std())

print("lpips bic_u = ",bic_lpips_u.mean(),bic_lpips_u.std())
print("lpips bic_v = ", bic_lpips_v.mean(),bic_lpips_v.std())

print("lpips model_u = ",model_lpips_u.mean(),model_lpips_u.std())
print("lpips model_v = ",model_lpips_v.mean(),model_lpips_v.std())


print("# - FSIM - #################################################")

print("FSIM LR_u = ",LR_FSIM_u.mean(),LR_FSIM_u.std())
print("FSIM LR_v = ",LR_FSIM_v.mean(),LR_FSIM_v.std())

print("FSIM bilinear_u = ",bilinear_FSIM_u.mean(),bilinear_FSIM_u.std())
print("FSIM bilinear_v = ",bilinear_FSIM_v.mean(),bilinear_FSIM_v.std())

print("FSIM bic_u = ",bic_FSIM_u.mean(),bic_FSIM_u.std())
print("FSIM bic_v = ",bic_FSIM_v.mean(),bic_FSIM_v.std())

print("FSIM model_u = ",model_FSIM_u.mean(),model_FSIM_u.std())
print("FSIM model_v = ",model_FSIM_v.mean(),model_FSIM_v.std())



print("# - EPI - #################################################")

print("EPI LR_u = ",LR_EPI_u.mean(),LR_EPI_u.std())
print("EPI LR_v = ",LR_EPI_v.mean(),LR_EPI_v.std())

print("EPI bilinear_u = ",bilinear_EPI_u.mean(),bilinear_EPI_u.std())
print("EPI bilinear_v = ",bilinear_EPI_v.mean(),bilinear_EPI_v.std())

print("EPI bic_u = ",bic_EPI_u.mean(),bic_EPI_u.std())
print("EPI bic_v = ",bic_EPI_v.mean(),bic_EPI_v.std())

print("EPI model_u = ",model_EPI_u.mean(),model_EPI_u.std())
print("EPI model_v = ",model_EPI_v.mean(),model_EPI_v.std())


print("# - EPI - #################################################")

print("TKE LR_u = ",LR_TKE_u.mean(),LR_TKE_u.std())
print("TKE LR_v = ",LR_TKE_v.mean(),LR_TKE_v.std())

print("TKE bilinear_u = ",bilinear_TKE_u.mean(),bilinear_TKE_u.std())
print("TKE bilinear_v = ",bilinear_TKE_v.mean(),bilinear_TKE_v.std())

print("TKE bic_u = ",bic_TKE_u.mean(),bic_TKE_u.std())
print("TKE bic_v = ",bic_TKE_v.mean(),bic_TKE_v.std())

print("TKE model_u = ",model_TKE_u.mean(),model_TKE_u.std())
print("TKE model_v = ",model_TKE_v.mean(),model_TKE_v.std())





### HISTOGRAMS FOR PRESENTATION  #################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#==================================================================================================
loss_data_u = {
    "MSE": (LR_MSE_u, model_MSE_u),
    "SSIM": (LR_SSIM_u, model_SSIM_u),
    "PSNR": (LR_PSNR_u, model_PSNR_u),
    "lpips": (LR_lpips_u, model_lpips_u),
    "FSIM": (LR_FSIM_u, model_FSIM_u),
    "EPI": (LR_EPI_u, model_EPI_u),
    "TKE": (LR_TKE_u, model_TKE_u),
}

for i, (data1_u, data4_u) in loss_data_u.items():

    # Plot histograms for LR and Model
    plt.figure(figsize=(10, 6))
    sns.histplot(data1_u, label=f"LR (μ={np.mean(data1_u):.5f}, σ={np.std(data1_u):.5f})", color="blue", kde=True, bins=30, alpha=0.5)
    sns.histplot(data4_u, label=f"Model (μ={np.mean(data4_u):.5f}, σ={np.std(data4_u):.5f})", color="gray", kde=True, bins=30, alpha=0.5)

    # Customize the plot
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{i} Histogram_u')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(f"Histograms/His/{i}-histogram_u.png", dpi=300)

    # Close the figure to free memory
    plt.close()

#==================================================================================================
loss_data_v = {
    "MSE": (LR_MSE_v, model_MSE_v),
    "SSIM": (LR_SSIM_v, model_SSIM_v),
    "PSNR": (LR_PSNR_v, model_PSNR_v),
    "lpips": (LR_lpips_v, model_lpips_v),
    "FSIM": (LR_FSIM_v, model_FSIM_v),
    "EPI": (LR_EPI_v, model_EPI_v),
    "TKE": (LR_TKE_v, model_TKE_v),
}

for i, (data1_v, data4_v) in loss_data_v.items():

    # Plot histograms for LR and Model
    plt.figure(figsize=(10, 6))
    sns.histplot(data1_v, label=f"LR (μ={np.mean(data1_v):.5f}, σ={np.std(data1_v):.5f})", color="blue", kde=True, bins=30, alpha=0.5)
    sns.histplot(data4_v, label=f"Model (μ={np.mean(data4_v):.5f}, σ={np.std(data4_v):.5f})", color="gray", kde=True, bins=30, alpha=0.5)

    # Customize the plot
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{i} Histogram_v')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(f"Histograms/His/{i}-histogram_v.png", dpi=300)

    # Close the figure to free memory
    plt.close()


exit()



#   Histograms ===================================================================================

import seaborn as sns
from scipy.stats import norm
#==================================================================================================
loss_data_u = {
    "MSE": (LR_MSE_u, bilinear_MSE_u, bic_MSE_u, model_MSE_u),
    "SSIM": (LR_SSIM_u, bilinear_SSIM_u, bic_SSIM_u, model_SSIM_u),
    "PSNR": (LR_PSNR_u, bilinear_PSNR_u, bic_PSNR_u, model_PSNR_u),
    "lpips": (LR_lpips_u, bilinear_lpips_u, bic_lpips_u, model_lpips_u),
    "FSIM": (LR_FSIM_u, bilinear_FSIM_u, bic_FSIM_u, model_FSIM_u),
    "EPI": (LR_EPI_u, bilinear_EPI_u, bic_EPI_u, model_EPI_u),
    "TKE": (LR_TKE_u, bilinear_TKE_u, bic_TKE_u, model_TKE_u),

}

for i, (data1_u, data2_u, data3_u, data4_u) in loss_data_u.items():

    # Create a range for the x-axis (using only data4_u)
    x = np.linspace(
        min(data4_u) - 1, 
        max(data4_u) + 1, 
        300
    )

    # Plot histogram for data4_u only
    plt.figure(figsize=(10, 6))
    sns.histplot(data4_u, label=f"Model (μ={np.mean(data4_u):.5f}, σ={np.std(data4_u):.5f})", color="gray", kde=True, bins=30, alpha=0.5)

    # Customize the plot
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{i} Histogram_u')
    plt.legend()
    plt.grid(True)

    # Save with dynamic filename
    plt.savefig("Histograms/His/"f"{i}-histogram_u.png", dpi=300)
   
   
    # Close the figure to free memory
    plt.close()


#========================================================================================================================
loss_data_v = {
    "MSE": (LR_MSE_v, bilinear_MSE_v, bic_MSE_v, model_MSE_v),
    "SSIM": (LR_SSIM_v, bilinear_SSIM_v, bic_SSIM_v, model_SSIM_v),
    "PSNR": (LR_PSNR_v, bilinear_PSNR_v, bic_PSNR_v, model_PSNR_v),
    "lpips": (LR_lpips_v, bilinear_lpips_v, bic_lpips_v, model_lpips_v),
    "FSIM": (LR_FSIM_v, bilinear_FSIM_v, bic_FSIM_v, model_FSIM_v),
    "EPI": (LR_EPI_v, bilinear_EPI_v, bic_EPI_v, model_EPI_v),
    "TKE": (LR_TKE_v, bilinear_TKE_v, bic_TKE_v, model_TKE_v),

}

for i, (data1_v, data2_v, data3_v, data4_v) in loss_data_v.items():

    # Create a range for the x-axis (using only data4_v)
    x = np.linspace(
        min(data4_v) - 1, 
        max(data4_v) + 1, 
        300
    )

    # Plot histogram for data4_v only
    plt.figure(figsize=(10, 6))
    sns.histplot(data4_v, label=f"Model (μ={np.mean(data4_v):.5f}, σ={np.std(data4_v):.5f})", color="gray", kde=True, bins=30, alpha=0.5)

    # Customize the plot
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{i} Histogram_v')
    plt.legend()
    plt.grid(True)

    # Save with dynamic filename
    plt.savefig("Histograms/PRUSR_His/"f"{i}-histogram_v.png", dpi=300)
 
    # Close the figure to free memory
    plt.close()


    

#   Histograms ===================================================================================

# import seaborn as sns
# from scipy.stats import norm
#==================================================================================================
loss_data_u = {
    "MSE": (LR_MSE_u, bilinear_MSE_u, bic_MSE_u, model_MSE_u),
    "SSIM": (LR_SSIM_u, bilinear_SSIM_u, bic_SSIM_u, model_SSIM_u),
    "PSNR": (LR_PSNR_u, bilinear_PSNR_u, bic_PSNR_u, model_PSNR_u),
    "lpips": (LR_lpips_u, bilinear_lpips_u, bic_lpips_u, model_lpips_u),
    "FSIM": (LR_FSIM_u, bilinear_FSIM_u, bic_FSIM_u, model_FSIM_u),
    "EPI": (LR_EPI_u, bilinear_EPI_u, bic_EPI_u, model_EPI_u),
    "TKE": (LR_TKE_u, bilinear_TKE_u, bic_TKE_u, model_TKE_u),

}

for i, (data1_u, data2_u, data3_u, data4_u) in loss_data_u.items():

    stats = {
        "LR": (np.mean(data1_u), np.std(data1_u)),
        "Bilinear": (np.mean(data2_u), np.std(data2_u)),
        "Bicubic": (np.mean(data3_u), np.std(data3_u)),
        "Model": (np.mean(data4_u), np.std(data4_u)),
    }

    # Create a range for the x-axis
    x = np.linspace(
        min(np.concatenate([data1_u, data2_u, data3_u, data4_u])) - 1, 
        max(np.concatenate([data1_u, data2_u, data3_u, data4_u])) + 1, 
        300
    )

     # Plot histograms
    plt.figure(figsize=(10, 6))
    colors = ["blue", "green", "red", "gray"]
    
    for (label, (mean, std)), color, data in zip(stats.items(), colors, [data1_u, data2_u, data3_u, data4_u]):
        sns.histplot(data, label=f"{label} (μ={mean:.5f}, σ={std:.5f})", color=color, kde=True, bins=30, alpha=0.5)

    # Customize the plot
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{i} Histogram_u')
    plt.legend()
    plt.grid(True)

    # Save with dynamic filename
    plt.savefig("Histograms/His/"f"{i}-histograms_u.png", dpi=300)
    
    # Close the figure to free memory
    plt.close()

#========================================================================================================================
loss_data_v = {
    "MSE": (LR_MSE_v, bilinear_MSE_v, bic_MSE_v, model_MSE_v),
    "SSIM": (LR_SSIM_v, bilinear_SSIM_v, bic_SSIM_v, model_SSIM_v),
    "PSNR": (LR_PSNR_v, bilinear_PSNR_v, bic_PSNR_v, model_PSNR_v),
    "lpips": (LR_lpips_v, bilinear_lpips_v, bic_lpips_v, model_lpips_v),
    "FSIM": (LR_FSIM_v, bilinear_FSIM_v, bic_FSIM_v, model_FSIM_v),
    "EPI": (LR_EPI_v, bilinear_EPI_v, bic_EPI_v, model_EPI_v),
    "TKE": (LR_TKE_v, bilinear_TKE_v, bic_TKE_v, model_TKE_v),

}

for i, (data1_v, data2_v, data3_v, data4_v) in loss_data_v.items():

    stats = {
        "LR": (np.mean(data1_v), np.std(data1_v)),
        "Bilinear": (np.mean(data2_v), np.std(data2_v)),
        "Bicubic": (np.mean(data3_v), np.std(data3_v)),
        "Model": (np.mean(data4_v), np.std(data4_v)),
    }

    # Create a range for the x-axis
    x = np.linspace(
        min(np.concatenate([data1_v, data2_v, data3_v, data4_v])) - 1, 
        max(np.concatenate([data1_v, data2_v, data3_v, data4_v])) + 1, 
        300
    )

     # Plot histograms
    plt.figure(figsize=(10, 6))
    colors = ["blue", "green", "red", "gray"]
    
    for (label, (mean, std)), color, data in zip(stats.items(), colors, [data1_v, data2_v, data3_v, data4_v]):
        sns.histplot(data, label=f"{label} (μ={mean:.5f}, σ={std:.5f})", color=color, kde=True, bins=30, alpha=0.5)

    # Customize the plot
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{i} Histogram_v')
    plt.legend()
    plt.grid(True)

    # Save with dynamic filename
    plt.savefig("Histograms/His/"f"{i}-histograms_v.png", dpi=300)
    
    # Close the figure to free memory
    plt.close()



 