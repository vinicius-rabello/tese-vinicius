import torch

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE=3e-4
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=500
ROOT_FOLDER="models/PRUSR/"