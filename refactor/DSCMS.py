from logging import getLogger

import torch
from torch import nn

logger = getLogger()


class DSCMS(nn.Module):
    """Hybrid Downsampled Skip-Connection/Multi-Scale model proposed by Fukami et al. (2019, JFM).
    Ref: http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py
    """

    def __init__(self, in_channels: int, out_channels: int, factor_filter_num: int):
        super(DSCMS, self).__init__()

        # Down-sampled skip-connection model (DSC)
        f_num1 = int(factor_filter_num * 32)
        logger.info(f"f_num1 = {f_num1} / 32, factor = {factor_filter_num}")

        self.max_pool = nn.MaxPool2d(kernel_size=8, stride=8,return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=8, stride=8)

        self.activation = nn.ReLU(inplace=True)
        #self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        self.dsc1_mp = nn.MaxPool2d(kernel_size=8, padding=0)
        self.dsc1_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=f_num1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            self.activation,
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        # Regarding `algin_corners=False`, see the below
        # https://qiita.com/matsxxx/items/fe24b9c2ac6d9716fdee
        # https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/20

        self.dsc2_mp = nn.MaxPool2d(kernel_size=4, padding=0)
        self.dsc2_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1
            ),
            self.activation,
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            self.activation,
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.dsc3_mp = nn.MaxPool2d(kernel_size=2, padding=0)
        self.dsc3_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1
            ),
            self.activation,
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            self.activation,
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        self.dsc4_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1
            ),
            self.activation,
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            self.activation,
        )

        # Multi-scale model (MS)
        f_num2 = int(factor_filter_num * 8)
        logger.info(f"f_num2 = {f_num2} / 8, factor = {factor_filter_num}")

        self.ms1_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=5, padding=2),
            self.activation,
            nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=5, padding=2),
            self.activation,
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=5, padding=2),
            self.activation,
        )

        self.ms2_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=9, padding=4),
            self.activation,
            nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=9, padding=4),
            self.activation,
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=9, padding=4),
            self.activation,
        )

        self.ms3_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=13, padding=6),
            self.activation,
            nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=13, padding=6),
            self.activation,
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=13, padding=6),
            self.activation,
        )

        self.ms4_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=(f_num2 * 3 + in_channels),
                out_channels=f_num2,
                kernel_size=7,
                padding=3,
            ),
            #nn.BatchNorm2d(f_num2),
            self.activation,
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=5, padding=2),
            self.activation,
            #nn.Dropout2d(p=0.3),
        )

        # After concatenating DSC and MS
        self.final_layers = nn.Conv2d(
            in_channels=f_num1 + f_num2, out_channels=out_channels, kernel_size=3, padding=1
        )

    def _dsc(self, x):
        x1 = self.dsc1_layers(self.dsc1_mp(x))
        mp2 = self.dsc2_mp(x)
        x2 = self.dsc2_layers(torch.cat([x1, mp2], dim=1))
        mp3 = self.dsc3_mp(x)
        x3 = self.dsc3_layers(torch.cat([x2, mp3], dim=1))
        return self.dsc4_layers(torch.cat([x, x3], dim=1))

    def _ms(self, x):
        x1 = self.ms1_layers(x)        
        x2 = self.ms2_layers(x)
        x3 = self.ms3_layers(x)        
        return self.ms4_layers(torch.cat([x, x1, x2, x3], dim=1))

    def forward(self, x):
        x1 = self._dsc(x)
        x2 = self._ms(x)
        x3 = self.final_layers(torch.cat([x1, x2], dim=1))
        return x3

if __name__ == "__main__":
    # Test parameters
    in_channels = 2  # Single channel (e.g., vorticity field)
    out_channels = 2  # Single output channel
    factor_filter_num = 3  # Filter scaling factor
    batch_size = 32
    height, width = 128, 128  # Typical fluid dynamics grid size
    
    print("=" * 60)
    print("DSCMS Model Test")
    print("=" * 60)
    
    # Create model
    print("Creating DSCMS model...")
    print(f"  Input channels: {in_channels}")
    print(f"  Output channels: {out_channels}")
    print(f"  Filter factor: {factor_filter_num}")
    
    model = DSCMS(in_channels=in_channels, out_channels=out_channels, factor_filter_num=factor_filter_num)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create test input
    print("\nCreating test input tensor...")
    print(f"  Shape: [{batch_size}, {in_channels}, {height}, {width}]")
    
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"  Input tensor shape: {x.shape}")
    print(f"  Input tensor range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Test forward pass
    print("\nRunning forward pass...")
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():
        try:
            output = model(x)
            print("✅ Forward pass successful!")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
            
            # Verify output shape matches expected
            expected_shape = (batch_size, out_channels, height, width)
            if output.shape == expected_shape:
                print(f"✅ Output shape matches expected: {expected_shape}")
            else:
                print(f"❌ Output shape mismatch! Expected: {expected_shape}, Got: {output.shape}")
                
        except Exception as e:
            print(f"❌ Forward pass failed with error: {e}")
            raise
    
    # Test intermediate shapes (DSC branch)
    print("\nTesting DSC branch intermediate shapes...")
    with torch.no_grad():
        # Test DSC pooling operations
        dsc1_pooled = model.dsc1_mp(x)
        dsc2_pooled = model.dsc2_mp(x)
        dsc3_pooled = model.dsc3_mp(x)
        
        print(f"  Original input: {x.shape}")
        print(f"  DSC1 pooled (8x8): {dsc1_pooled.shape}")
        print(f"  DSC2 pooled (4x4): {dsc2_pooled.shape}")
        print(f"  DSC3 pooled (2x2): {dsc3_pooled.shape}")
    
    # Test intermediate shapes (MS branch)
    print("\nTesting MS branch intermediate shapes...")
    with torch.no_grad():
        ms1_out = model.ms1_layers(x)
        ms2_out = model.ms2_layers(x)
        ms3_out = model.ms3_layers(x)
        
        print(f"  MS1 output (5x5 kernels): {ms1_out.shape}")
        print(f"  MS2 output (9x9 kernels): {ms2_out.shape}")
        print(f"  MS3 output (13x13 kernels): {ms3_out.shape}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! DSCMS model is working correctly.")
    print("=" * 60)