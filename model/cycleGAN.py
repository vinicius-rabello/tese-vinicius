import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from DSCMS import DSCMS
from loaderGan import SuperResNpyDataset2
from torch.utils.data import Dataset, DataLoader,random_split
##########################################################################
torch.autograd.set_detect_anomaly(True)
refinementLevel = 4
Nx,Ny = 128,128
##########################################################################
################################################################################
def upscale_with_bicubic(x):
    return F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)  # bicubic interpolation (smoother)
##########################################################################
class Upsampling(nn.Module):
    def __init__(self):
        super(Upsampling, self).__init__()

    def forward(self, image, p, q):
        return image.repeat_interleave(p, dim=2).repeat_interleave(q, dim=3)     # nearest-neighbor (blocky, no interpolation)
##########################################################################
##########################################################################
class GeneratorG2(nn.Module):
    def __init__(self,input_channels=3,output_channels=3,alpha=0.2):  # RGB,RGB,LeakyReLU activation function's negative slope

        super(GeneratorG, self).__init__()
        self.DSCMS = DSCMS(2,2,1)

    def forward(self, x):
        x = upscale_with_bicubic(x)
        x = self.DSCMS(x)
        return x        

##########################################################################
# Accepts low Res and gives high Res output
class GeneratorG(nn.Module):
    def __init__(self,input_channels=3,output_channels=3,alpha=0.2): # RGB,RGB,LeakyReLU activation function's negative slope
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

    def forward(self, x):                              #[batch, 2, 128, 128]# → [batch, 2, 256, 256]# → [batch, 2, 512, 512]
                                                                           
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
        x = self.fc2(x)                    # Fully Connected (FC) Layer

        return x
##########################################################################
class GANModel(nn.Module):
    def __init__(self, input_channels, learning_rate):   
        super(GANModel, self).__init__()  # Initialize nn.Module

        ###  Combines both generators (G and F) and discriminators (X and Y)

        self.generator_G = GeneratorG(input_channels=input_channels,output_channels=input_channels)        
        self.generator_F = GeneratorF(input_channels=input_channels,output_channels=input_channels)
        self.discriminator_X = DiscriminatorX(input_channels=input_channels)
        self.discriminator_Y = DiscriminatorY(input_channels=input_channels)
        
        self.learning_rate = learning_rate

        # ✅ Add this line  Only for L1
        self.identity_weight = 5.0  # Weight for identity loss


        ###  Sets up optimizers for all four networks
        
        beta1 = 0.0
        self.optimizer_G = optim.Adam(self.generator_G.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        self.optimizer_F = optim.Adam(self.generator_F.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        self.optimizer_DX = optim.Adam(self.discriminator_X.parameters(), lr=learning_rate*3.0, betas=(beta1, 0.999))
        self.optimizer_DY = optim.Adam(self.discriminator_Y.parameters(), lr=learning_rate*3.0, betas=(beta1, 0.999))

    #-#-#  gradient penalty function to stabilize training (used in WGAN-GP)
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
    #----------------------------------------------------------------------------------------
    #  Original train ============================================================================

    # def train(self, X, Y):
    #     # Generator forward pass
    #     #noise = torch.randn_like(X) * 0.1
    #     #X = X + noise

    # # -----------------------------
    # # === 1. Train Discriminators ===
    # # -----------------------------

    #     Y_pred = self.generator_G(X)                                # G(X)     # X → Y
    #     X_pred = self.generator_F(Y)                                # F(Y)     # Y → X
    #     Y_pred2 = self.generator_G(X_pred)                          # G(F(Y))  # Y → X → Y (cycle)
    #     X_pred2 = self.generator_F(Y_pred)                          # F(G(X))  # X → Y → X (cycle)

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
    #     self.optimizer_DX.zero_grad()                       # 1. Clear old gradients for discriminator_X   
    #     DX_loss.backward()                                  # 2. Compute gradients (backpropagation) for DX_loss
    #     self.optimizer_DX.step()                            # 3. Update weights of discriminator_X using optimizer
    #     #---
    #     self.optimizer_DY.zero_grad()                       # 1. Clear old gradients for discriminator_Y
    #     DY_loss.backward()
    #     self.optimizer_DY.step()

    # # -----------------------------
    # # === 2. Train Generator G === (X → Y)
    # # -----------------------------

       
    #     Y_pred = self.generator_G(X)
    #     X_pred = self.generator_F(Y)
    #     Y_pred2 = self.generator_G(X_pred)     # G(F(Y))
    #     X_pred2 = self.generator_F(Y_pred)     # F(G(X))
    #     DY_fake = self.discriminator_Y(Y_pred)
    #     DX_fake = self.discriminator_X(X_pred)

    #     cycle_X_loss = F.mse_loss(X_pred2, X)
    #     cycle_Y_loss = F.mse_loss(Y_pred2, Y)
    #     cycle_loss = cycle_X_loss + cycle_Y_loss


    #     G_loss = -DY_fake.mean() + 10.0 * cycle_loss
    #     F_loss = -DX_fake.mean() + 10.0 * cycle_loss
    #     # print("1", cycle_loss,G_loss,F_loss)  

    #     total_generator_loss = G_loss + F_loss
    #     self.optimizer_G.zero_grad()
    #     self.optimizer_F.zero_grad()
    #     total_generator_loss.backward()
    #     self.optimizer_G.step()
    #     self.optimizer_F.step()

    #     # self.optimizer_G.zero_grad()
    #     # G_loss.backward(retain_graph=True)
    #     # self.optimizer_G.step()

    #     # self.optimizer_F.zero_grad()
    #     # F_loss.backward(retain_graph=True)
    #     # self.optimizer_F.step()
        
    # # -----------------------------
    # # === 3. Train Generator F === (Y → X)
    # # -----------------------------

    #     # Y_pred = self.generator_G(X)
    #     # X_pred = self.generator_F(Y)
    #     # Y_pred2 = self.generator_G(X_pred)
    #     # X_pred2 = self.generator_F(Y_pred)
    #     # DY_fake = self.discriminator_Y(Y_pred)
    #     # DX_fake = self.discriminator_X(X_pred)

    #     # cycle_X_loss = F.mse_loss(X_pred2, X)
    #     # cycle_Y_loss = F.mse_loss(Y_pred2, Y)
    #     # cycle_loss = cycle_X_loss + cycle_Y_loss
        

    #     # G_loss = -DY_fake.mean() + 10.0 * cycle_loss
    #     # F_loss = -DX_fake.mean() + 10.0 * cycle_loss 
    #     # print("2", cycle_loss,G_loss,F_loss)   
        
    #     # self.optimizer_F.zero_grad()
    #     # F_loss.backward(retain_graph=True)
    #     # self.optimizer_F.step()



    #     losses = np.array([
    #         DX_loss.item(),
    #         DY_loss.item(),
    #         G_loss.item(),
    #         F_loss.item(),
    #         cycle_loss.item(),
    #         gp_X.item(),
    #         gp_Y.item()
    #     ])

    #     return losses
    
    ###########################################################################################

    ## NEW L1 loss and identity loss

    def train(self, X, Y):
        # -----------------------------
        # === 1. Train Discriminators ===
        # -----------------------------
        Y_pred = self.generator_G(X)                                # G(X)
        X_pred = self.generator_F(Y)                                # F(Y)
        Y_pred2 = self.generator_G(X_pred)                          # G(F(Y))
        X_pred2 = self.generator_F(Y_pred)                          # F(G(X))

        DX_real = self.discriminator_X(X)
        DX_fake = self.discriminator_X(X_pred.detach())
        DY_real = self.discriminator_Y(Y)
        DY_fake = self.discriminator_Y(Y_pred.detach())

        gp_X = self.gradient_penalty(X, X_pred, self.discriminator_X)
        gp_Y = self.gradient_penalty(Y, Y_pred, self.discriminator_Y)

        DX_loss = DX_fake.mean() - DX_real.mean() + 10.0 * gp_X
        DY_loss = DY_fake.mean() - DY_real.mean() + 10.0 * gp_Y

        self.optimizer_DX.zero_grad()
        DX_loss.backward()
        self.optimizer_DX.step()

        self.optimizer_DY.zero_grad()
        DY_loss.backward()
        self.optimizer_DY.step()

        # -----------------------------
        # === 2. Train Generators ===
        # -----------------------------

        Y_pred = self.generator_G(X)
        X_pred = self.generator_F(Y)
        Y_pred2 = self.generator_G(X_pred)
        X_pred2 = self.generator_F(Y_pred)
        DY_fake = self.discriminator_Y(Y_pred)
        DX_fake = self.discriminator_X(X_pred)

        # === Cycle consistency loss using L1 ===
        cycle_X_loss = F.l1_loss(X_pred2, X)
        cycle_Y_loss = F.l1_loss(Y_pred2, Y)
        cycle_loss = cycle_X_loss + cycle_Y_loss

        # === Identity loss ===
        identity_Y = self.generator_G(Y)
        identity_X = self.generator_F(X)

        identity_Y_resized = F.interpolate(identity_Y, size=Y.shape[2:], mode='bilinear', align_corners=False)
        identity_X_resized = F.interpolate(identity_X, size=X.shape[2:], mode='bilinear', align_corners=False)
        # print(f"identity_Y: {identity_Y_resized.shape}, Y: {Y.shape}")
        # print(f"identity_X: {identity_X_resized.shape}, X: {X.shape}")
        identity_Y = identity_Y_resized
        identity_X = identity_X_resized
        # exit()

        identity_loss = F.l1_loss(identity_Y, Y) + F.l1_loss(identity_X, X)
        identity_loss = self.identity_weight * identity_loss  # Define this in __init__

        # === Generator losses ===
        G_loss = -DY_fake.mean() + 10.0 * cycle_loss + identity_loss
        F_loss = -DX_fake.mean() + 10.0 * cycle_loss + identity_loss

        total_generator_loss = G_loss + F_loss
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()
        total_generator_loss.backward()
        self.optimizer_G.step()
        self.optimizer_F.step()

        # === Log and return ===
        losses = np.array([
            DX_loss.item(),
            DY_loss.item(),
            G_loss.item(),
            F_loss.item(),
            cycle_loss.item(),
            identity_loss.item(),
            gp_X.item(),
            gp_Y.item()
        ])

        return losses





    #############################################################################################
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
#Train model
def train_gan(model, data_loader, num_epochs, learning_rate=1e-4, device='cuda'):
    model.to(device)
    loss = np.zeros(8)
    for epoch in range(num_epochs):
        for batch_idx, (X, Y) in enumerate(data_loader):
            X, Y = X.to(device), Y.to(device)
            batchLoss = model.train(X, Y)
            loss += batchLoss
            # ------------------- Train Discriminators -------------------

        loss = loss / len(dataLoader)
        # print(f"Epoch [{epoch+1}/{num_epochs}] | D_X Loss: {loss[0]:.4f} | D_Y Loss: {loss[1]:.4f} | G Loss: {loss[2]:.4f} | F Loss: {loss[3]:.4f} | C Loss: {loss[4]:.4f} | GX {loss[5]:.4f} | GY {loss[6]:.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] | D_X Loss: {loss[0]:.4f} | D_Y Loss: {loss[1]:.4f} | "
      f"G Loss: {loss[2]:.4f} | F Loss: {loss[3]:.4f} | C Loss: {loss[4]:.4f} | "
      f"GX: {loss[5]:.4f} | GY: {loss[6]:.4f} | Identity: {loss[7]:.4f}")
        
        if (epoch + 1) % 1 == 0:  # Save model checkpoints every 10 epochs
            torch.save(model.state_dict(), f'../weights/cycleGan-L1-HLH/gan2_{epoch+1}.pth')




##############################################################################################################



##############################################################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GANModel(input_channels = 2,learning_rate=1.0e-4)
#model = torch.nn.DataParallel(model)
model = model.to(device)

model.generator_F.eval()
model.generator_G.eval()

#################################################################################################
# Testing Inference

# x = torch.randn(1,2,int(Nx/refinementLevel), int(Ny/refinementLevel)).to(device)  # Example input
# y = torch.randn(1,2,Nx, Ny).to(device)  # Example input

# output1 = model.forward(x,direction='X2Y')
# print("X2Y",output1.shape)  # Should be [1, 2, 320, 544]
# output2 = model.forward(y,direction='Y2X')
# print("Y2X",output2.shape)  # Should be [1, 2, 320, 544]
# output3 = model.forward(x,direction='DX')
# print("DX",output3.shape)  # Should be [1, 2, 320, 544]
# output4 = model.forward(y,direction='DY')
# print("DY",output4.shape)  # Should be [1, 2, 320, 544]
##########################################################################
import os
import random


class LowResDomainDataset(Dataset):
    def __init__(self, data_folder, lr_files, skip=0):
        data_list = []
        for f in lr_files:
            data = np.load(os.path.join(data_folder, f))[skip:]
            data_list.append(data)
        self.lr_data = np.concatenate(data_list, axis=0)

        # Use only 2 channels (as before)
        self.lr_data = self.lr_data[:, :, :, 2:4]

        # Normalize (optional, or move to training script)
        mean = np.mean(self.lr_data, axis=(0, 1, 2), keepdims=True)
        std = np.std(self.lr_data, axis=(0, 1, 2), keepdims=True)
        self.lr_data = (self.lr_data - mean) / (std + 1e-8)

    def __len__(self):
        return self.lr_data.shape[0]

    def __getitem__(self, idx):
        x = self.lr_data[idx]
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        return x

class HighResDomainDataset(Dataset):
    def __init__(self, data_folder, hr_files, skip=0):
        data_list = []
        for f in hr_files:
            data = np.load(os.path.join(data_folder, f))[skip:]
            data_list.append(data)
        self.hr_data = np.concatenate(data_list, axis=0)

        # Use only 2 channels
        self.hr_data = self.hr_data[:, :, :, 2:4]

        # Normalize (optional)
        mean = np.mean(self.hr_data, axis=(0, 1, 2), keepdims=True)
        std = np.std(self.hr_data, axis=(0, 1, 2), keepdims=True)
        self.hr_data = (self.hr_data - mean) / (std + 1e-8)

    def __len__(self):
        return self.hr_data.shape[0]

    def __getitem__(self, idx):
        y = self.hr_data[idx]
        y = torch.tensor(y, dtype=torch.float32).permute(2, 0, 1)
        return y
    
class UnpairedSuperResDataset(Dataset):
    def __init__(self, lr_dataset, hr_dataset):
        self.lr_dataset = lr_dataset
        self.hr_dataset = hr_dataset

    def __len__(self):
        # We'll just base length on the larger of the two
        return max(len(self.lr_dataset), len(self.hr_dataset))

    def __getitem__(self, idx):
        # Randomly sample from each domain
        lr_idx = random.randint(0, len(self.lr_dataset) - 1)
        hr_idx = random.randint(0, len(self.hr_dataset) - 1)

        lr = self.lr_dataset[lr_idx]
        hr = self.hr_dataset[hr_idx]

        return lr, hr


##########################################################################
# Dataset Loading

data_folder = "../data"

lr_files = ["25/window_2003.npy","25/window_2004.npy","25/window_2005.npy","25/window_2006.npy"]
hr_files = ["100/window_2003.npy","100/window_2004.npy","100/window_2005.npy","100/window_2006.npy"]
#lr_files = ["25/window_2003.npy"]
#hr_files = ["100/window_2003.npy"]


## unpair   ########################################################################
# Create datasets (unpaired)
lr_dataset = LowResDomainDataset(data_folder, lr_files)
hr_dataset = HighResDomainDataset(data_folder, hr_files)

# Create loaders
lr_loader = DataLoader(lr_dataset, batch_size=16, shuffle=True)
hr_loader = DataLoader(hr_dataset, batch_size=16, shuffle=True)


# Wrap them into an unpaired dataset
unpaired_dataset = UnpairedSuperResDataset(lr_dataset, hr_dataset)

# Now create your DataLoader
dataLoader = DataLoader(unpaired_dataset, batch_size=16, shuffle=True)
##########################################################################  


# Pair     #########################################################################

# dataset = SuperResNpyDataset2(data_folder, lr_files, hr_files)
# dataLoader = DataLoader(dataset, batch_size=16, shuffle=True)

##########################################################################

# # Define split sizes
train_size = int(0.8 * len(unpaired_dataset))
val_size = len(unpaired_dataset) - train_size

# # Split the dataset
train_dataset, val_dataset = random_split(unpaired_dataset, [train_size, val_size])

# # Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
##########################################################################
# Running Training
#model.load_state_dict(torch.load('../weights/gan_model_epoch_448.pth'))
# model.load_state_dict(torch.load('../weights/cycleGan-LH/gan2_280.pth'))
train_gan(model, dataLoader, 1000, learning_rate=1e-5, device=device)
##########################################################################








