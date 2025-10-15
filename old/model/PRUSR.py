from logging import getLogger
import torch.nn.functional as F

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy.ndimage import zoom
########################################################################################


logger = getLogger()


  ## PRUSR HAMID ------------------------------------------------------------------------

#####################################################################################################
# Define the Pre-activation Residual Block (PRB) class

# class PRB(nn.Module):
#     def __init__(self, in_channels, out_channels, num_repeats, kernel_size=3, stride=1, padding=1, negative_slope=0.01):
#         super(PRB, self).__init__()

#         layers = []
#         #for i in range(num_repeats):
#         #    layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
#         #    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
#         #    in_channels = out_channels  # Ensure output channels match for subsequent layers

#            # Construct a sequence of residual blocks with LeakyReLU activation
#         layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
#         layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
#         layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
#         layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False))
#         layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
#         layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False))

#         self.conv_layers = nn.Sequential(*layers)

#     def forward(self, x):
#         skip_connection = x  # Save input for skip connection
#         out = self.conv_layers(x)
#         return out + skip_connection  # Skip connection added
# #####################################################################################################


# class PRUSR(nn.Module):
#     """Hybrid Downsampled Skip-Connection/Multi-Scale model proposed by Fukami et al. (2019, JFM).
#     Ref: http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py
#     """

#     def __init__(self, in_channels: int, out_channels: int, factor_filter_num: int):
#         super(PRUSR, self).__init__()

#         # Down-sampled skip-connection model (DSC)
#         f_num1 = int(factor_filter_num * 32)
#         logger.info(f"f_num1 = {f_num1} / 32, factor = {factor_filter_num}")

#         self.max_pool = nn.MaxPool2d(kernel_size=8, stride=8,return_indices=True)
#         self.unpool = nn.MaxUnpool2d(kernel_size=8, stride=8)
#         self.dsc_PRB = PRB(f_num1, f_num1, 3, kernel_size=3, stride=1, padding=1, negative_slope=0.01)

#         #self.activation = nn.ReLU(inplace=True)
#         self.activation = nn.LeakyReLU(negative_slope=0.01)
#         #---------------
#         self.dsc1_mp = nn.Sequential(
#             self.activation,
#             nn.MaxPool2d(kernel_size=8, padding=0),
#             #nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=8, stride=8, padding=0)
#         )
#         self.dsc1_layers = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=in_channels,out_channels=f_num1,kernel_size=4,stride=2,padding=1),
#             PRB(in_channels=f_num1, out_channels=f_num1,num_repeats=3, kernel_size=3, stride=1, negative_slope=0.01),
#         )
#         #---------------
#         self.dsc2_mp = nn.Sequential(
#             self.activation,
#             nn.MaxPool2d(kernel_size=4, padding=0),
#         )
#         self.dsc2_layers = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=in_channels+f_num1,out_channels=f_num1,kernel_size=4,stride=2,padding=1),
#             PRB(in_channels=f_num1, out_channels=f_num1,num_repeats=3, kernel_size=3, stride=1, negative_slope=0.01),
#         )
#         #---------------
#         self.dsc3_mp = nn.Sequential(
#             self.activation,
#             nn.MaxPool2d(kernel_size=2, padding=0),
#         )
#         self.dsc3_layers = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=in_channels+f_num1,out_channels=f_num1,kernel_size=4,stride=2,padding=1),
#             PRB(in_channels=f_num1, out_channels=f_num1,num_repeats=3, kernel_size=3, stride=1, negative_slope=0.01),
#         )
#         #---------------
#         self.dsc4_layers = nn.Sequential(
#             self.activation,
#             nn.Conv2d(
#                 in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1
#             ),
            
#             PRB(in_channels=f_num1, out_channels=f_num1,num_repeats=3, kernel_size=3, stride=1, negative_slope=0.01),             
#         )
#         #---------------
#         # Multi-scale model (MS)
#         f_num2 = int(factor_filter_num * 8)
#         logger.info(f"f_num2 = {f_num2} / 8, factor = {factor_filter_num}")

#         self.ms1_layers = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=5, padding=2),
#             #self.activation,
#             #nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=5, padding=2),
#             #self.activation,
#             #nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=5, padding=2),
#             #self.activation,
#             PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=5, stride=1,padding=2, negative_slope=0.01),  
#             PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=5, stride=1,padding=2, negative_slope=0.01),  
#         )

#         self.ms2_layers = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=9, padding=4),
#             #self.activation,
#             #nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=9, padding=4),
#             #self.activation,
#             #nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=9, padding=4),
#             #self.activation,
#             PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=9, stride=1,padding=4, negative_slope=0.01),  
#             PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=9, stride=1,padding=4, negative_slope=0.01),  
#         )

#         self.ms3_layers = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=13, padding=6),
#             #self.activation,
#             #nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=13, padding=6),
#             #self.activation,
#             #nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=13, padding=6),
#             #self.activation,
#             PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=13, stride=1,padding=6, negative_slope=0.01),  
#             PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=13, stride=1,padding=6, negative_slope=0.01),  
#         )

#         self.ms4_layers = nn.Sequential(
#             self.activation,
#             nn.Conv2d(
#                 in_channels=(2*f_num2 * 3 + in_channels),
#                 out_channels=f_num2,
#                 kernel_size=7,
#                 padding=3,
#             ),
#             #nn.BatchNorm2d(f_num2),            
#             #nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=5, padding=2),
#             #self.activation,
#             #nn.Dropout2d(p=0.3),
#             PRB(in_channels=f_num2, out_channels=f_num2,num_repeats=3, kernel_size=13, stride=1,padding=6, negative_slope=0.01),  
#             PRB(in_channels=f_num2, out_channels=f_num2,num_repeats=3, kernel_size=13, stride=1,padding=6, negative_slope=0.01),  
#         )

#         # After concatenating DSC and MS
#         self.final_layers = nn.Sequential(
#             self.activation,
#             nn.Conv2d(in_channels=f_num1 + f_num2, out_channels=out_channels, kernel_size=3, padding=1),
#         )

#     def _dsc(self, x):
#         x1 = self.dsc1_layers(self.dsc1_mp(x))
#         mp2 = self.dsc2_mp(x)
#         x2 = self.dsc2_layers(torch.cat([x1, mp2], dim=1))
#         mp3 = self.dsc3_mp(x)
#         x3 = self.dsc3_layers(torch.cat([x2, mp3], dim=1))
#         return self.dsc4_layers(torch.cat([x, x3], dim=1))

#     def _ms(self, x):
#         x1 = self.ms1_layers(x)        
#         x2 = self.ms2_layers(x)
#         x3 = self.ms3_layers(x)        
#         return self.ms4_layers(torch.cat([x, x1, x2, x3], dim=1))

#     def forward(self, x):
#         x1 = self._dsc(x)
#         x2 = self._ms(x)
#         x3 = self.final_layers(torch.cat([x1, x2], dim=1))
#         return x3
########################################################################################

  ## PRUSR new ------------------------------------------------------------------------

########################################################################################
class PRB(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeats, kernel_size=3, stride=1, padding=1, negative_slope=0.01):
        super(PRB, self).__init__()

        layers = []
        #for i in range(num_repeats):
        #    layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        #    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        #    in_channels = out_channels  # Ensure output channels match for subsequent layers

           # Construct a sequence of residual blocks with LeakyReLU activation
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False))
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        skip_connection = x  # Save input for skip connection
        out = self.conv_layers(x)
        return out + skip_connection  # Skip connection added
#####################################################################################################


class PRUSR(nn.Module):
    """Hybrid Downsampled Skip-Connection/Multi-Scale model proposed by Fukami et al. (2019, JFM).
    Ref: http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py
    """

    def __init__(self, in_channels: int, out_channels: int, factor_filter_num: int):
        super(PRUSR, self).__init__()

        # Down-sampled skip-connection model (DSC)
        f_num1 = int(factor_filter_num * 32)
        logger.info(f"f_num1 = {f_num1} / 32, factor = {factor_filter_num}")

        #----------------------------------------------------------------------------------------------

        self.max_pool = nn.MaxPool2d(kernel_size=8, stride=8,return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=8, stride=8)
        self.dsc_PRB = PRB(f_num1, f_num1, 3, kernel_size=3, stride=1, padding=1, negative_slope=0.01)
        #self.activation = nn.ReLU(inplace=True)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        #----------------------------------------------------------------------------------

        self.dsc1_mp = nn.Sequential(
            # self.activation,
            # nn.MaxPool2d(kernel_size=8, padding=0),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=8, stride=8, padding=0),
            self.activation,

        )
        self.dsc1_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,out_channels=f_num1,kernel_size=4,stride=2,padding=1),
            PRB(in_channels=f_num1, out_channels=f_num1,num_repeats=3, kernel_size=3, stride=1, negative_slope=0.01),
        )
        #---------------
        self.dsc2_mp = nn.Sequential(
            # self.activation,
            # nn.MaxPool2d(kernel_size=4, padding=0),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=4, padding=0),
            self.activation,

        )
        self.dsc2_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels+f_num1,out_channels=f_num1,kernel_size=4,stride=2,padding=1),
            PRB(in_channels=f_num1, out_channels=f_num1,num_repeats=3, kernel_size=3, stride=1, negative_slope=0.01),
        )
        #---------------
        self.dsc3_mp = nn.Sequential(
            # self.activation,
            # nn.MaxPool2d(kernel_size=2, padding=0),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2, padding=0),
            self.activation,

        )
        self.dsc3_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels+f_num1,out_channels=f_num1,kernel_size=4,stride=2,padding=1),
            PRB(in_channels=f_num1, out_channels=f_num1,num_repeats=3, kernel_size=3, stride=1, negative_slope=0.01),
        )
        #---------------
        self.dsc4_layers = nn.Sequential(
            # self.activation,
            nn.Conv2d(in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            self.activation,

            PRB(in_channels=f_num1, out_channels=f_num1,num_repeats=3, kernel_size=3, stride=1, negative_slope=0.01),             
        )
        #---------------
        # Multi-scale model (MS)
        f_num2 = int(factor_filter_num * 8)
        logger.info(f"f_num2 = {f_num2} / 8, factor = {factor_filter_num}")

        self.ms1_layers = nn.Sequential(
            # self.activation,               # --- Not good when apply before conv  -------------------
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=5, padding=2),
            self.activation,

            PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=5, stride=1,padding=2, negative_slope=0.01),  
            PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=5, stride=1,padding=2, negative_slope=0.01),
            PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=5, stride=1,padding=2, negative_slope=0.01),  
  
        )

        self.ms2_layers = nn.Sequential(
            # self.activation,
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=9, padding=4),
            self.activation,

            PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=9, stride=1,padding=4, negative_slope=0.01),  
            PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=9, stride=1,padding=4, negative_slope=0.01),
            PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=9, stride=1,padding=4, negative_slope=0.01),  
  
        )

        self.ms3_layers = nn.Sequential(
            # self.activation,                    
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=13, padding=6),
            self.activation,

            PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=13, stride=1,padding=6, negative_slope=0.01),
            PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=13, stride=1,padding=6, negative_slope=0.01),    
            PRB(in_channels=2*f_num2, out_channels=2*f_num2,num_repeats=3, kernel_size=13, stride=1,padding=6, negative_slope=0.01),  
        )

        self.ms4_layers = nn.Sequential(
            self.activation,
            nn.Conv2d(in_channels=(2*f_num2 * 3 + in_channels), out_channels=f_num2, kernel_size=7, padding=3),

            PRB(in_channels=f_num2, out_channels=f_num2,num_repeats=3, kernel_size=13, stride=1,padding=6, negative_slope=0.01),  
            PRB(in_channels=f_num2, out_channels=f_num2,num_repeats=3, kernel_size=13, stride=1,padding=6, negative_slope=0.01),  
        )

        # After concatenating DSC and MS
        self.final_layers = nn.Sequential(
            # self.activation,
            nn.Conv2d(in_channels=f_num1 + f_num2, out_channels=out_channels, kernel_size=3, padding=1),
            self.activation,
        )

    def _dsc(self, x):
        x1 = self.dsc1_layers(self.dsc1_mp(x))
        mp2 = self.dsc2_mp(x)
        x2 = self.dsc2_layers(torch.cat([x1, mp2], dim=1))
        mp3 = self.dsc3_mp(x)
        x3 = self.dsc3_layers(torch.cat([x2, mp3], dim=1))
        return self.dsc4_layers(torch.cat([x, x3], dim=1))

    def _ms(self, x):
        # x = self.activation(x)  # Apply activation before layers
        x1 = self.ms1_layers(x)        
        x2 = self.ms2_layers(x)
        x3 = self.ms3_layers(x)        
        return self.ms4_layers(torch.cat([x, x1, x2, x3], dim=1))

    def forward(self, x):
        x1 = self._dsc(x)
        x2 = self._ms(x)
        x3 = self.final_layers(torch.cat([x1, x2], dim=1))
        return x3