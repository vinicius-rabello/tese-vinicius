from logging import getLogger

import torch
from torch import nn

logger = getLogger()


class SRCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SRCNN, self).__init__()

        self.activation = nn.ReLU()
        self.f1 = 9
        self.f2 = 5
        self.f3 = 5
        self.n1 = 64
        self.n2 = 32
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=self.n1, kernel_size=self.f1, padding=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.n1, out_channels=self.n2, kernel_size=self.f2, padding=2
        )  

        self.conv3 = nn.Conv2d(
            in_channels=self.n2, out_channels=out_channels, kernel_size=self.f3, padding=2
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.activation(x1)
        x2 = self.conv2(x1)
        x2 = self.activation(x2)
        x3 = self.conv3(x2)
        return x3

def test():
    # Test parameters
    in_channels = 2
    out_channels = 2
    batch_size = 32
    height, width = 128, 128
    x = torch.randn(batch_size, in_channels, height, width)
    model = SRCNN(in_channels=in_channels, out_channels=out_channels)
    pred = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")

if __name__ == "__main__":
    test()