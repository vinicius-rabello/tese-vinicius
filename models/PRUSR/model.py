from logging import getLogger

import torch
from torch import nn


logger = getLogger()

class PRB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, negative_slope=0.2):
        super(PRB, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same', bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same', bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same', bias=False)
        )

    def forward(self, x):
        skip_connection = x
        out = self.conv_layers(x)
        return out + skip_connection


class PRUSR(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_filter_num: int = 3):
        super(PRUSR, self).__init__()

        self.activation = nn.LeakyReLU(negative_slope=0.2)

        # PS Path
        self.ps1_layers = nn.Sequential(
            self.activation,
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, padding='same'),
            PRB(in_channels=16, out_channels=16, kernel_size=5, stride=1),  
            PRB(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            PRB(in_channels=16, out_channels=16, kernel_size=5, stride=1),  
  
        )

        self.ps2_layers = nn.Sequential(
            self.activation,
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=9, padding='same'),
            PRB(in_channels=16, out_channels=16, kernel_size=9, stride=1),  
            PRB(in_channels=16, out_channels=16, kernel_size=9, stride=1),
            PRB(in_channels=16, out_channels=16, kernel_size=9, stride=1),  
  
        )

        self.ps3_layers = nn.Sequential(
            self.activation,                    
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=13, padding='same'),
            PRB(in_channels=16, out_channels=16, kernel_size=13, stride=1),
            PRB(in_channels=16, out_channels=16, kernel_size=13, stride=1),    
            PRB(in_channels=16, out_channels=16, kernel_size=13, stride=1),  
        )

        self.ps_final_layers = nn.Sequential(
            self.activation,
            nn.Conv2d(in_channels=in_channels + 3*16, out_channels=16, kernel_size=9, padding='same'),
            PRB(in_channels=16, out_channels=16, kernel_size=9, stride=1),  
            PRB(in_channels=16, out_channels=16, kernel_size=9, stride=1),  
        )

        # U-path
        self.enc1_layers = nn.Sequential(
            self.activation,
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding='same')
        )
        
        self.enc2_layers = nn.Sequential(
            self.activation,
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=1)
        )

        self.enc3_layers = nn.Sequential(
            self.activation,
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=4, padding=1)
        )

        self.enc4_layers = nn.Sequential(
            self.activation,
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=8, padding=1)
        )

        self.dec1_layers = nn.Sequential(
            PRB(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            self.activation,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        )

        self.dec2_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            PRB(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        )

        self.dec3_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            PRB(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        )

        self.dec4_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            PRB(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        )

        # After concatenating PS and U-PATH
        self.final_layers = nn.Sequential(
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, padding='same'),
        )

    def _ps(self, x):
        x1 = self.ps1_layers(x)        
        x2 = self.ps2_layers(x)
        x3 = self.ps3_layers(x)  
        return self.ps_final_layers(torch.cat([x, x1, x2, x3], dim=1))
    
    def _u_path(self, x):
        enc1 = self.enc1_layers(x)
        enc2 = self.enc2_layers(x)
        enc3 = self.enc3_layers(x)
        enc4 = self.enc4_layers(x)
        dec4 = self.dec4_layers(enc4)
        dec3 = self.dec3_layers(dec4 + enc3)
        dec2 = self.dec2_layers(dec3 + enc2)
        dec1 = self.dec1_layers(dec2 + enc1)
        return dec1

    def forward(self, x):
        x1 = self._ps(x)
        x2 = self._u_path(x)
        x3 = self.final_layers(torch.cat([x1, x2], dim=1))
        return x3
    

def test():
    # Test parameters
    in_channels = 2  # Single channel (e.g., vorticity field)
    out_channels = 2  # Single output channel
    batch_size = 32
    height, width = 128, 128  # Typical fluid dynamics grid size
    x = torch.randn(batch_size, in_channels, height, width)
    model = PRUSR(in_channels=in_channels, out_channels=out_channels)
    pred = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")

if __name__ == "__main__":
    test()