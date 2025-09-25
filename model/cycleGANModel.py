import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from loaderGan import SuperResNpyDataset2
from torch.utils.data import Dataset, DataLoader,random_split
from DSCMS import DSCMS

##########################################################################
torch.autograd.set_detect_anomaly(True)
refinementLevel = 4
Nx,Ny = 128,128
##########################################################################
################################################################################
def upscale_with_bicubic(x):
    return F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
##########################################################################
class Upsampling(nn.Module):
    def __init__(self):
        super(Upsampling, self).__init__()

    def forward(self, image, p, q):
        return image.repeat_interleave(p, dim=2).repeat_interleave(q, dim=3)
##########################################################################
##########################################################################
class GeneratorG2(nn.Module):
    def __init__(self,input_channels=3,output_channels=3,alpha=0.2):
        super(GeneratorG, self).__init__()
        self.DSCMS = DSCMS(2,2,1)

    def forward(self, x):
        x = upscale_with_bicubic(x)
        x = self.DSCMS(x)
        return x        

##########################################################################
# Accepts low Res and gives high Res output
class GeneratorG(nn.Module):
    def __init__(self,input_channels=3,output_channels=3,alpha=0.2):
        super(GeneratorG, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, output_channels, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU(negative_slope=alpha)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        #self.upsample = Upsampling()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        #x = self.upsample(x, 2, 2)
        x = self.upsample(x)

        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        #x = self.upsample(x, 2, 2)
        x = self.upsample(x)

        x = self.activation(self.conv5(x))
        x = self.conv6(x)

        return x        
##########################################################################
class GeneratorF(nn.Module):
    def __init__(self,input_channels=3,output_channels=3,alpha=0.2):
        super(GeneratorF, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=alpha)

        #self.downsample = F.avg_pool2d(kernel_size=2)
        self.downsample = nn.MaxPool2d(kernel_size=2, padding=0)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.downsample(x)

        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.downsample(x)

        x = self.activation(self.conv5(x))
        x = self.conv6(x)

        return x
##########################################################################
class DiscriminatorX(nn.Module):
    def __init__(self,input_channels=3,alpha=0.2):
        super(DiscriminatorX, self).__init__()
        K = [32, 64, 64, 128, 128, 256, 256]

        self.conv1 = nn.Conv2d(input_channels, K[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(K[0], K[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(K[1], K[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(K[2], K[3], kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(K[3], K[4], kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(K[4], K[5], kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(K[5], K[6], kernel_size=3, padding=1)

        self.fc1 = nn.Linear(K[6] * int(Nx/8/refinementLevel) * int(Ny/8/refinementLevel), 256)  # Assuming input size of 256x256
        self.fc2 = nn.Linear(256, 1)
        self.activation = nn.LeakyReLU(negative_slope=alpha)

        self.downsample = nn.MaxPool2d(kernel_size=2, padding=0)
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.downsample(x)

        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.downsample(x)

        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))
        x = self.downsample(x)

        x = self.activation(self.conv7(x))
        x = x.view(x.size(0), -1)
        
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x
##########################################################################
class DiscriminatorY(nn.Module):
    def __init__(self,input_channels=3,alpha=0.2):
        super(DiscriminatorY, self).__init__()
        K = [16, 32, 32, 64, 64, 128, 128, 256, 256]

        self.conv1 = nn.Conv2d(input_channels, K[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(K[0], K[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(K[1], K[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(K[2], K[3], kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(K[3], K[4], kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(K[4], K[5], kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(K[5], K[6], kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(K[6], K[7], kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(K[7], K[8], kernel_size=3, padding=1)

        self.fc1 = nn.Linear(K[8] * int(Nx/16) * int(Ny/16), 256)
        self.fc2 = nn.Linear(256, 1)
        self.activation = nn.LeakyReLU(negative_slope=alpha)
        
        self.downsample = nn.MaxPool2d(kernel_size=2, padding=0)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.downsample(x)

        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.downsample(x)

        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))
        x = self.downsample(x)

        x = self.activation(self.conv7(x))
        x = self.activation(self.conv8(x))
        x = self.downsample(x)

        x = self.activation(self.conv9(x))
        x = x.view(x.size(0), -1)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x
##########################################################################
class GANModel(nn.Module):
    def __init__(self, input_channels, learning_rate):   
        super(GANModel, self).__init__()  # Initialize nn.Module

        self.generator_G = GeneratorG(input_channels=input_channels,output_channels=input_channels)        
        self.generator_F = GeneratorF(input_channels=input_channels,output_channels=input_channels)
        self.discriminator_X = DiscriminatorX(input_channels=input_channels)
        self.discriminator_Y = DiscriminatorY(input_channels=input_channels)
        
        self.learning_rate = learning_rate
        
        beta1 = 0.0
        self.optimizer_G = optim.Adam(self.generator_G.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        self.optimizer_F = optim.Adam(self.generator_F.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        self.optimizer_DX = optim.Adam(self.discriminator_X.parameters(), lr=learning_rate*2.0, betas=(beta1, 0.999))
        self.optimizer_DY = optim.Adam(self.discriminator_Y.parameters(), lr=learning_rate*2.0, betas=(beta1, 0.999))
    #--------------------
    def gradient_penalty(self, real, fake, discriminator):
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device)
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)

        d_interpolated = discriminator(interpolated)

        gradients = autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        penalty = torch.mean((gradient_norm - 1) ** 2)

        return penalty
    #--------------------

    # def train(self, X, Y):
    #     # Generator forward pass
    #     Y_pred = self.generator_G(X)
    #     X_pred = self.generator_F(Y)
    #     Y_pred2 = self.generator_G(X_pred)
    #     X_pred2 = self.generator_F(Y_pred)

    #     # Discriminator forward pass
    #     DX_real = self.discriminator_X(X)
    #     DX_fake = self.discriminator_X(X_pred.detach())

    #     DY_real = self.discriminator_Y(Y)
    #     DY_fake = self.discriminator_Y(Y_pred.detach())
        
    #     # Gradient penalty
    #     gp_X = self.gradient_penalty(X, X_pred, self.discriminator_X)
    #     gp_Y = self.gradient_penalty(Y, Y_pred, self.discriminator_Y)

    #     # Loss calculations
    #     DX_loss = DX_fake.mean() - DX_real.mean() + 10.0 * gp_X
    #     DY_loss = DY_fake.mean() - DY_real.mean() + 10.0 * gp_Y
                
    #     # Optimizer steps
    #     self.optimizer_DX.zero_grad()
    #     DX_loss.backward()
    #     self.optimizer_DX.step()
    #     #---
    #     self.optimizer_DY.zero_grad()
    #     DY_loss.backward()
    #     self.optimizer_DY.step()

    #     #---
    #     Y_pred = self.generator_G(X)
    #     X_pred = self.generator_F(Y)
    #     Y_pred2 = self.generator_G(X_pred)
    #     X_pred2 = self.generator_F(Y_pred)
    #     DY_fake = self.discriminator_Y(Y_pred)
    #     DX_fake = self.discriminator_X(X_pred)
    #     cycle_X_loss = F.mse_loss(X_pred2, X)
    #     cycle_Y_loss = F.mse_loss(Y_pred2, Y)
    #     cycle_loss = cycle_X_loss + cycle_Y_loss

    #     G_loss = -DY_fake.mean() + 10.0 * cycle_loss
    #     F_loss = -DX_fake.mean() + 10.0 * cycle_loss

    #     self.optimizer_G.zero_grad()
    #     G_loss.backward(retain_graph=True)
    #     self.optimizer_G.step()
        
    #     #---
    #     Y_pred = self.generator_G(X)
    #     X_pred = self.generator_F(Y)
    #     Y_pred2 = self.generator_G(X_pred)
    #     X_pred2 = self.generator_F(Y_pred)
    #     DY_fake = self.discriminator_Y(Y_pred)
    #     DX_fake = self.discriminator_X(X_pred)
    #     cycle_X_loss = F.mse_loss(X_pred2, X)
    #     cycle_Y_loss = F.mse_loss(Y_pred2, Y)
    #     cycle_loss = cycle_X_loss + cycle_Y_loss

    #     G_loss = -DY_fake.mean() + 10.0 * cycle_loss
    #     F_loss = -DX_fake.mean() + 10.0 * cycle_loss        
        
    #     self.optimizer_F.zero_grad()
    #     F_loss.backward(retain_graph=True)
    #     self.optimizer_F.step()

    #     losses = np.array([
    #         DX_loss.item(),
    #         DY_loss.item(),
    #         G_loss.item(),
    #         F_loss.item(),
    #         cycle_loss.item()
    #     ])

    #     return losses
    #---------------------
    def forward(self, x, direction='X2Y'):
        """
        Forward pass for inference.
        
        Args:
            x (torch.Tensor): The input tensor.
            direction (str): 'X2Y' for X -> Y transformation using generator_G,
                             'Y2X' for Y -> X transformation using generator_F.
                             
        Returns:
            torch.Tensor: The generated output.
        """
        if direction == 'X2Y':
            return self.generator_G(x)
        elif direction == 'Y2X':
            return self.generator_F(x)
        elif direction == "DX":
            return self.discriminator_X(x)
        elif direction == "DY":
            return self.discriminator_Y(x)
        else:
            raise ValueError("Invalid direction. Choose either 'X2Y' or 'Y2X'.")    
##########################################################################
