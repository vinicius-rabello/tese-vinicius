import torch

# =============================================================================
# DEVICE
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# DATA
# =============================================================================

HR_FILES  = ['data/100/window_2003.npy']
LR_SIZE   = 8
HR_SIZE   = 32
VAL_SPLIT = 0.2

# --- normalisation — set mean=0 / std=1 to disable ---
INP_MEAN = [0.0, 0.0]
INP_STD  = [1.0, 1.0]
GT_MEAN  = [0.0, 0.0]
GT_STD   = [1.0, 1.0]

# =============================================================================
# MODEL
# =============================================================================

N_RESBLOCKS  = 16
N_FEATS      = 64
RES_SCALE    = 1.0
WIDTH        = 256
T            = 2        # number of SoFoNO blocks  (cf. SRNO's 'blocks')
LOCAL_BRANCH = 'Conv'   # 'Conv' or 'Attn'
INIT_S       = 0        # initial value of the learnable gate scalar s
RANGES       = None     # coordinate range tuple (min, max); None → (-1, 1)

# =============================================================================
# TRAINING
# =============================================================================

BATCH_SIZE    = 16
NUM_WORKERS   = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 500
WARMUP_EPOCHS = 50
EPOCH_SAVE    = 50      # save a numbered checkpoint every N epochs

# =============================================================================
# PATHS
# =============================================================================

ROOT_FOLDER = 'models/sofono'