import torch

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE=4e-5
BATCH_SIZE=16
NUM_WORKERS=4
NUM_EPOCHS=1050
ROOT_FOLDER="models/SoFoNO/"
