import torch

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE=2e-4
BATCH_SIZE=32
NUM_WORKERS=4
NUM_EPOCHS=10
LAMBDA_CYCLE=10
LAMBDA_IDENTITY=0.0