import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

HR_FILES = ['data/100/window_2003.npy']
LR_SIZE = 8
HR_SIZE = 32
VAL_SPLIT = 0.2

# --- normalisation ---
INP_MEAN = [0.0, 0.0]
INP_STD  = [1.0, 1.0]
GT_MEAN  = [0.0, 0.0]
GT_STD   = [1.0, 1.0]

# --- model ---
N_RESBLOCKS = 16
N_FEATS     = 64
RES_SCALE   = 1.0
WIDTH       = 256
BLOCKS      = 16

# --- training ---
LEARNING_RATE = 3e-4
BATCH_SIZE    = 1
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 500
WARMUP_EPOCHS = 50
EPOCH_SAVE    = 50

# --- system ---
NUM_WORKERS = 4  # default sugerido

# --- paths ---
ROOT_FOLDER = './models/srno'