import torch

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE=2e-4
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=250
ROOT_FOLDER="models/DSCMS/"