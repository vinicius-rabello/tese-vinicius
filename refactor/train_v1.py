import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.image import PeakSignalNoiseRatio
import lpips

from DSCMS import DSCMS
from loader import SuperResNpyDataset2

def check_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient Norm: {total_norm}")

data_folder = "./data"

# Explicitly provide lists of file names
lr_files = ["25/window_2003.npy"]#,"25/window_2004.npy","25/window_2005.npy","25/window_2006.npy"]
hr_files = ["100/window_2003.npy"]#,"100/window_2004.npy","100/window_2005.npy","100/window_2006.npy"]

dataset = SuperResNpyDataset2(data_folder, lr_files, hr_files)

# Define split sizes
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DSCMS(in_channels=2, out_channels=2, factor_filter_num=3)
model = torch.nn.DataParallel(model)
model = model.to(device)

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

print("total trainig size = ", len(train_loader), "   total validation size = ", len(val_loader) )

# Hyperparameters
num_epochs = 25#00
learning_rate = 1.0e-4

psnr_metric = PeakSignalNoiseRatio(data_range=1).to(device)  # Ensure PSNR is on the same device
lpips_loss = lpips.LPIPS(net='vgg').to(device)

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))



class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer=8):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:layer]  # Extract first few layers
        self.vgg = vgg.eval().to(device)  # Move to GPU
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights
        self.criterion = nn.L1Loss()  # L1 loss for feature maps

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return self.criterion(x_features, y_features)

vgg_loss = VGGPerceptualLoss().to(device)  # Initialize VGG perceptual loss

lpips_weight = 1.0
vgg_weight = 0.01  # Small weight for perceptual loss

# Loss and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Training Loops
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)
        lr_np = lr.cpu().numpy()
        lr = torch.tensor(lr_np).to(device)
        lr_upsampled = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)

        # Forward pass
        outputs = model(lr_upsampled)
        outputs = outputs.to(device)

        output = outputs.squeeze().detach().cpu().numpy()  # shape: (2, H, W)
        high = hr.squeeze().cpu().numpy()                    # shape: (2, H, W)

        loss = criterion(outputs, hr)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        epoch_loss += loss.item()


    avg_loss = epoch_loss / len(train_loader)

    # ---- Validation Phase ----
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    mse_loss = 0
    NER = 0
    with torch.no_grad():  # No gradient computation in validation
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            lr_np = lr.cpu().numpy()
            lr = torch.tensor(lr_np).to(device)
            lr_upsampled = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)

            outputs = model(lr_upsampled)
            outputs = outputs.to(device)
            loss = criterion(outputs, hr)

            val_loss += loss.item()

            l2_norm = torch.norm(hr, p=2)
            NER += loss / l2_norm

    # val_loss = val_loss / len(val_loader)    
    mse_loss =  mse_loss / len(val_loader)    
    NER = NER / len(val_loader)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss Train.: {avg_loss:.6f}, Loss Val.: {val_loss:.6f}, , NER: {NER:.6f}, Learning rate: {current_lr}")
    with open("output_PRUSR.txt", "a") as f:
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss Train.: {avg_loss:.4f}, Loss Val.: {val_loss:.4f}, Loss Bi. {mse_loss:.4f}, , NER: {NER:.6f},Learning rate: {current_lr}\n")


    # Save model checkpoint every 10 epochs
    if (epoch+1) % 2 == 1:
        torch.save(model.state_dict(), f'test_{epoch+1}.pth')
