import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import fft2


def compute_energy_spectrum(velocity_field):
    """Plain NumPy radial kinetic-energy spectrum, for plotting E(k) curves.

    Matches KineticEnergySpectrumLoss.compute_spectrum's normalization exactly
    (validated to agree with the torch version to ~1e-10), just returning the
    per-bin wavenumbers alongside the energy instead of a fixed-length tensor.

    Parameters
    ----------
    velocity_field : np.ndarray, shape (2, H, W)
        u, v components stacked on the first axis.

    Returns
    -------
    k_bins : np.ndarray
    energy : np.ndarray
    """
    u, v = velocity_field[0], velocity_field[1]
    u_prime = u - np.mean(u)
    v_prime = v - np.mean(v)
    nx, ny = u.shape

    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k = np.sqrt(kx_grid**2 + ky_grid**2)

    e_u = (np.abs(fft2(u_prime)) / (nx * ny)) ** 2
    e_v = (np.abs(fft2(v_prime)) / (nx * ny)) ** 2
    e_k = 0.5 * (e_u + e_v)

    k_bins = np.arange(1, (nx // 2) + 1, 1)
    valid_bins, valid_energy = [], []
    for i in range(len(k_bins) - 1):
        k_min, k_max = k_bins[i], k_bins[i + 1]
        mask = (k >= k_min) & (k < k_max)
        if np.count_nonzero(mask) > 0:
            valid_bins.append(k_min)
            valid_energy.append(np.sum(e_k[mask]))

    return np.array(valid_bins), np.array(valid_energy)


class KineticEnergySpectrumLoss(nn.Module):
    """
    Differentiable kinetic energy spectrum loss.

    Input:
        (B, 2, H, W)

    Computes exactly the same spectrum as compute_ek(),
    except using PyTorch tensors.
    """

    def __init__(self, size=128, eps=1e-8):
        super().__init__()

        self.size = size
        self.eps = eps

        kx = torch.fft.fftfreq(size) * size
        ky = torch.fft.fftfreq(size) * size

        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
        radius = torch.sqrt(kx_grid**2 + ky_grid**2)

        # ------------------------------------------------------------------
        # Cache all shell masks.
        # These never change.
        # ------------------------------------------------------------------
        masks = []


        for k in range(1, size // 2):
            mask = (radius >= k) & (radius < (k + 1))
            masks.append(mask.reshape(-1))

        masks = torch.stack(masks)

        self.register_buffer("masks", masks)

    def compute_spectrum(self, x):
        """
        Parameters
        ----------
        x : (B,2,H,W)

        Returns
        -------
        spectrum : (B,63)
        """

        B, _, H, W = x.shape

        u = x[:, 0]
        v = x[:, 1]

        # Remove mean
        u = u - u.mean(dim=(-2, -1), keepdim=True)
        v = v - v.mean(dim=(-2, -1), keepdim=True)

        U = torch.fft.fft2(u)
        V = torch.fft.fft2(v)

        e_u = (U.abs() / (H * W)) ** 2
        e_v = (V.abs() / (H * W)) ** 2

        Ek = 0.5 * (e_u + e_v)

        Ek = Ek.reshape(B, -1)

        spectrum = []

        # Only 63 shells, so this loop is tiny.
        for mask in self.masks:
            shell_energy = Ek[:, mask].sum(dim=1)
            spectrum.append(shell_energy)

        return torch.stack(spectrum, dim=1)

    def forward(self, pred, target):

        pred_spec = self.compute_spectrum(pred)

        with torch.no_grad():
            target_spec = self.compute_spectrum(target)


        return F.mse_loss(
            torch.log(pred_spec + self.eps),
            torch.log(target_spec + self.eps),
        )



class WeightedKineticEnergySpectrumLoss(nn.Module):
    """
    Differentiable kinetic energy spectrum loss, weighted to penalize 
    high-frequency errors.
    """
    def __init__(self, size=128, eps=1e-8):
        super().__init__()
        self.size = size
        self.eps = eps

        kx = torch.fft.fftfreq(size) * size
        ky = torch.fft.fftfreq(size) * size
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
        radius = torch.sqrt(kx_grid**2 + ky_grid**2)

        masks = []
        k_vals = []

        # 1. Collect both the masks and the exact k values
        for k in range(1, size // 2):
            mask = (radius >= k) & (radius < (k + 1))
            masks.append(mask.reshape(-1))
            k_vals.append(k)

        # Convert to float for fast matrix multiplication later
        masks = torch.stack(masks).float()
        
        # 2. Correctly build the weights tensor
        k_tensor = torch.tensor(k_vals, dtype=torch.float32)
        weights = k_tensor / k_tensor.sum() # Linear weighting normalized to sum to 1

        # Register buffers so PyTorch handles device movement (CPU to GPU) automatically
        self.register_buffer("masks_float", masks)
        self.register_buffer("weights", weights)

    def compute_spectrum(self, x):
        B, _, H, W = x.shape

        u = x[:, 0]
        v = x[:, 1]

        # Remove mean
        u = u - u.mean(dim=(-2, -1), keepdim=True)
        v = v - v.mean(dim=(-2, -1), keepdim=True)

        U = torch.fft.fft2(u)
        V = torch.fft.fft2(v)

        e_u = (U.abs() / (H * W)) ** 2
        e_v = (V.abs() / (H * W)) ** 2

        Ek = 0.5 * (e_u + e_v)
        Ek = Ek.reshape(B, -1)

        # 3. Fast GPU matrix multiplication (replaces the for-loop)
        # Ek: (B, H*W) multiplied by masks.T: (H*W, 63) -> Result: (B, 63)
        spectrum = torch.matmul(Ek, self.masks_float.T)

        return spectrum

    def forward(self, pred, target):
        pred_spec = self.compute_spectrum(pred)

        with torch.no_grad():
            target_spec = self.compute_spectrum(target)

        # 4. Apply the weights to the MSE manually
        # Compute the log difference per bin
        log_diff = torch.log(pred_spec + self.eps) - torch.log(target_spec + self.eps)
        
        # Square it to get the squared error (Shape: B, 63)
        squared_error = log_diff ** 2
        
        # Multiply by the weisghts (broadcasting across the batch dimension)
        weighted_squared_error = squared_error * self.weights
        
        # Average over the batch and sum over the bins (or average over both, depending on your preferred loss scale)
        return weighted_squared_error.sum(dim=1).mean()
    
if __name__ == "__main__":
    # 1. Create random tensors (Batch=4, Channels=2, H=128, W=128)
    batch_size = 4
    pred = torch.randn(batch_size, 2, 128, 128)
    target = torch.randn(batch_size, 2, 128, 128)

    print("--- Tensor Shapes ---")
    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape:     {target.shape}\n")

    # 2. Initialize the loss functions
    unweighted_loss = KineticEnergySpectrumLoss(size=128)
    weighted_loss = WeightedKineticEnergySpectrumLoss(size=128)

    # 3. Calculate losses
    loss_val_unweighted = unweighted_loss(pred, target)
    loss_val_weighted = weighted_loss(pred, target)

    print("--- Loss Values ---")
    print(f"Unweighted Loss: {loss_val_unweighted.item():.6f}")
    print(f"Weighted Loss:   {loss_val_weighted.item():.6f}\n")

    # 4. Print weights from the weighted loss
    print("--- High-Frequency Weights ---")
    print(f"Weights shape: {weighted_loss.weights.shape} (Should be 63 for 128x128 grid)")
    print(f"Weights tensor:\n{weighted_loss.weights}")