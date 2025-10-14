import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding_mode='reflect',
                **kwargs
            ) if down else nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, down=True, use_act=True, kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, channels, down=True, use_act=False, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, in_channels, num_features=64, num_residuals=6):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, down=True, use_act=True, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, down=True, use_act=True, kernel_size=3, stride=2, padding=1),
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, use_act=True, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features, down=False, use_act=True, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(
            num_features*1,
            in_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode='reflect'
        )
    
    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))
    
def test():
    # Test parameters
    in_channels = 2  # Single channel (e.g., vorticity field)
    out_channels = 2  # Single output channel
    batch_size = 32
    height, width = 128, 128  # Typical fluid dynamics grid size
    x = torch.randn(batch_size, in_channels, height, width)
    model = Generator(in_channels=in_channels)
    pred = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")

if __name__ == "__main__":
    test()