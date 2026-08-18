import torch

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE=3e-4
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=500
ROOT_FOLDER="models/DSCMS/test_composite_loss_mse_lpips_0.6_0.4/"

# --- NEW LOSS CONFIGURATION ---
# LOSS_MODE: 
# 1 = Pure MSE
# 2 = MSE + LPIPS
# 3 = MSE + LPIPS + Kinetic Energy Spectrum
LOSS_MODE = 2

# Define the weights for each mode. Ensure they sum to 1.0 for stability.
LOSS_WEIGHTS = {
    'mse': 0.6,
    'lpips': 0.4,
    'kinetic': 0.0
}