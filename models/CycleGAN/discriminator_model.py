import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_instance_norm=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode='reflect'
            ),
            nn.InstanceNorm2d(out_channels) if use_instance_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        layers=[]
        for i, feature in enumerate(features):
            layers.append(
                Block(
                    in_channels=in_channels,
                    out_channels=feature,
                    stride=1 if i==len(features)-1 else 2,
                    use_instance_norm=False if i==0 else True
                )
            )
            in_channels=feature
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=4,
                stride=1, 
                padding=1,
                padding_mode='reflect'
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))
    
def test():
    # Test parameters
    in_channels = 2  # Single channel (e.g., vorticity field)
    out_channels = 2  # Single output channel
    batch_size = 1
    height, width = 128, 128  # Typical fluid dynamics grid size
    x = torch.randn(batch_size, in_channels, height, width)
    model = Discriminator(in_channels=in_channels)
    pred = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")

if __name__ == "__main__":
    test()