import torch
import torch.nn.functional as F
import numpy as np
from scipy.fft import fft2

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