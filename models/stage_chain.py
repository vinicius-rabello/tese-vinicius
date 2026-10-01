"""Multistage chain for train.py (stage 4.7):

    y_hat(n) = y_hat(n-1)(x) + epsilon_n * f_n(x)

Every stage sees the original LR input x. `base` is the whole frozen chain up
to stage n-1 (a plain model for stage 1, a StageChain otherwise); only f_n is
trained. epsilon_n is the RMS of the residual of stage n-1 over the training
set, stored as a buffer so it travels with the checkpoint.

The original train_multistage.py / models/multistage.py stay untouched for the
old results; this is the version on the new run infrastructure.
"""

import torch
import torch.nn as nn


class StageChain(nn.Module):
    def __init__(self, base: nn.Module, stage: nn.Module, epsilon: float):
        super().__init__()
        self.base = base
        self.stage = stage
        self.register_buffer('epsilon', torch.tensor(float(epsilon)))
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

    def train(self, mode: bool = True):
        # model.train() must never put the frozen chain back in training mode
        # (SRNO/SoFoNO have normalization/dropout layers).
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            previous = self.base(x)
        return previous + self.epsilon * self.stage(x)
