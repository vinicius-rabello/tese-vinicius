########################################################################################
import torch
from torch.utils.data import Dataset, DataLoader,random_split
import numpy as np
from torch import nn
import os
import torch.nn.functional as F
from mixedLoss import CombinedLoss
from piq import  SSIMLoss
from torchmetrics.image import PeakSignalNoiseRatio
import lpips
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau





from DSCMS import DSCMS
from DSCMS_NN import DSCMS_NN
from DSCMS_PSS import DSCMS_PSS
from PRUSR import PRUSR
from CNN import CNN

from loader import SuperResNpyDataset,SuperResNpyDataset2D,SuperResNpyDataset2,SuperResNpyDataset3,SuperResNpyDataset42
from loader import SuperResNpyDatasetLH
########################################################################################
def check_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient Norm: {total_norm}")
########################################################################################
data_folder = "../data"

# Explicitly provide lists of file names
lr_files = ["25/window_2003.npy","25/window_2004.npy","25/window_2005.npy","25/window_2006.npy"]
hr_files = ["100/window_2003.npy","100/window_2004.npy","100/window_2005.npy","100/window_2006.npy"]
er = np.load("2D_error_maps/2D_error_maps_combined.npy")  

dataset = SuperResNpyDataset2D(data_folder, lr_files, hr_files,er)

print(np.shape(lr_files),np.shape(hr_files),len(dataset))


# Define split sizes ==============================================================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

print("total trainig size = ",len(train_loader),"   total validation size = ",len(val_loader) )
#########################################################################

##########################################################################################
# print(np.shape(errors),np.shape(errors))
# exit()

########################################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = DSCMS(2,2,3)
model = torch.nn.DataParallel(model)
model = model.to(device)


##########################################################################################
# def hook(module, input, output):
    # Check if the output is a tuple, which can happen in some layers
def hook(module, input, output):
    # Check if the output is a tuple, which can happen in some layers
    if isinstance(output, tuple):
        output = output[0]  # Extract the tensor if it's a tuple
    
    # Now check for NaN values in the tensor
    if torch.isnan(output).any():
        print(f"NaN detected in {module}")
        exit()

for name, module in model.named_modules():
    module.register_forward_hook(hook)
#==============================================================================================
    




# Hyperparameters
num_epochs = 2500
learning_rate = 1.0e-4

####################################################################
import torch.nn.functional as F
psnr_metric = PeakSignalNoiseRatio().to(device)  # Ensure PSNR is on the same device
######################################################################
# lpips_loss = lpips.LPIPS(net='vgg')  # Use 'alex' or 'squeeze' for different backbones

lpips_loss = lpips.LPIPS(net='vgg').to(device)
#################################################################################

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


#####################################################################
import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer=8):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:layer]  # Extract first few layers
        self.vgg = vgg.eval().cuda()  # Move to GPU
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights
        self.criterion = nn.L1Loss()  # L1 loss for feature maps

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return self.criterion(x_features, y_features)

vgg_loss = VGGPerceptualLoss().cuda()  # Initialize VGG perceptual loss

lpips_weight = 1.0
vgg_weight = 0.01  # Small weight for perceptual loss










## losses
# Loss and optimizer
criterion = nn.L1Loss()
# criterion = RMSELoss()
# criterion = nn.MSELoss()
#criterion = CombinedLoss()
# criterion = SSIMLoss(data_range=1.0)
# criterion = PeakSignalNoiseRatio(data_range=1.0)



optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

#model.load_state_dict(torch.load('../weights/CR_52.pth'))



counter = 0
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)


# Training Loops
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for lr, hr, er in train_loader:
        counter+=1
        # print(f"Epoch [{epoch+1}/{num_epochs}], Step [{counter}/{len(train_loader)}]")
        # lr, hr  = lr.to(device), hr.to(device) 

        lr, hr , er = lr.to(device), hr.to(device) ,er.to(device)
        # print(f"LR shape: {lr.shape}, HR shape: {hr.shape}","*********",np.shape(er))
        # exit()



        lr_np = lr.cpu().numpy()
        # lr_np = zoom(lr_np, zoom=(1, 1, 4, 4), order=3)

        lr = torch.tensor(lr_np).to(device)
        # lr = torch.tensor(lr_np, dtype=torch.float32).to(device)  # if it's not already tensor
        lr_upsampled = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
        # lr = lr.to(device)  # Move LR to device
        # print(np.shape(lr),np.shape(hr))
        # exit()
        
        ###############################################################################################################
        # lr = (lr - lr.min()) / (lr.max() - lr.min())
        # hr = (hr - hr.min()) / (hr.max() - hr.min())



        # Forward pass
        outputs = model(lr_upsampled)
        outputs = outputs.to(device)
        # outputs = torch.clamp(outputs, 0.0, 1.0)
        # er = torch.clamp(er, 0.0, 1.0)




        output = outputs.squeeze().detach().cpu().numpy()  # shape: (2, H, W)
        high = hr.squeeze().cpu().numpy()                    # shape: (2, H, W)


        ###  Normalization for lpips #######################################################################
        ###Convert 2-channel data to 3-channel (LPIPS expects 3-channel input)
        # if outputs.shape[1] == 2:
        #     mean_channel = outputs.mean(dim=1, keepdim=True)
        #     outputs = torch.cat([outputs, mean_channel], dim=1)
        #     hr = torch.cat([hr, hr.mean(dim=1, keepdim=True)], dim=1)






        # loss = lpips_loss(outputs, er).mean()

        loss = criterion(outputs, er)
 


        # # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        #check_gradient_norms(model)
        optimizer.step()
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)


        epoch_loss += loss.item()


    avg_loss = epoch_loss / len(train_loader)

    ####################################################################################
    ######################################################################################

    # ---- Validation Phase ----
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    mse_loss = 0
    NER = 0
    with torch.no_grad():  # No gradient computation in validation
        for lr, hr , er in val_loader:
            # lr, hr = lr.to(device), hr.to(device)
            lr, hr , er = lr.to(device), hr.to(device) ,er.to(device)

            lr_np = lr.cpu().numpy()
            # lr_np = zoom(lr_np, zoom=(1, 1, 4, 4), order=3)

            lr = torch.tensor(lr_np).to(device)
            lr_upsampled = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)



            outputs = model(lr_upsampled)
            outputs = outputs.to(device)
            # outputs = torch.clamp(outputs, 0.0, 1.0)
            # er = torch.clamp(er, 0.0, 1.0)




            # Convert 2-channel data to 3-channel (LPIPS expects 3-channel input)
            # if outputs.shape[1] == 2:
            #     mean_channel = outputs.mean(dim=1, keepdim=True)
            #     outputs = torch.cat([outputs, mean_channel], dim=1)
            #     hr = torch.cat([hr, hr.mean(dim=1, keepdim=True)], dim=1)


                # loss = lpips_loss(outputs, hr).mean()
            loss = criterion(outputs, er)



            val_loss += loss.item()

            l2_norm = torch.norm(hr, p=2)
            NER += loss / l2_norm

    # val_loss = val_loss / len(val_loader)    
    mse_loss = 100. * mse_loss / len(val_loader)    
    NER = NER / len(val_loader)

    # --------------------
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss Train.: {avg_loss:.6f}, Loss Val.: {val_loss:.6f}, Learning rate: {current_lr}")
    #print(f"Scaling Factor: {model.module.scaling_factor.item()}")
    with open("trashes/output_PRUSR.txt", "a") as f:
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss Train.: {avg_loss:.4f}, Loss Val.: {val_loss:.4f}, Loss Bi. {mse_loss:.4f}, Learning rate: {current_lr}\n")


    # Save model checkpoint every 10 epochs
    if (epoch) % 2 == 0:
        torch.save(model.state_dict(), f'../weights/2D_check2/Reza_{epoch}.pth')

########################################################################################
########################################################################################
########################################################################################
########################################################################################
