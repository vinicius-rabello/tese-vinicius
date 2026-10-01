"""Multistage residual model: a frozen stage-1 model plus a second network
trained to predict its leftover (normalized) error.

final(lr) = stage1(lr) + epsilon * stage2(lr)

stage1 is frozen (eval mode, requires_grad=False everywhere); only stage2's
parameters are ever updated. epsilon is a fixed scalar computed once, before
stage2 training starts, from the RMS of stage1's residual over the training
set (see train_multistage.py's compute_epsilon()).
"""

import torch
import torch.nn as nn


class MultistageModel(nn.Module):
    def __init__(self, stage1: nn.Module, stage2: nn.Module, epsilon: float):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.epsilon = epsilon

        for p in self.stage1.parameters():
            p.requires_grad = False
        self.stage1.eval()

    def forward(self, lr):
        with torch.no_grad():
            base = self.stage1(lr)
        return base + self.epsilon * self.stage2(lr)
