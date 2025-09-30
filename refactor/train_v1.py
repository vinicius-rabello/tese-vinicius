import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

model = DSCMS(in_channels=2, out_channels=2)
model = torch.nn.DataParallel(model)
model = model.to(device)

print("total trainig size = ", len(train_loader), "   total validation size = ", len(val_loader) )

# Hyperparameters
num_epochs = 2#00
learning_rate = 1.0e-4

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

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss Train.: {avg_loss:.6f}, Loss Val.: {val_loss:.6f}, Learning rate: {current_lr}")
    with open("output_PRUSR.txt", "a") as f:
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss Train.: {avg_loss:.4f}, Loss Val.: {val_loss:.4f},Learning rate: {current_lr}\n")


    # Save model checkpoint every 10 epochs
    if (epoch+1) % 2 == 1:
        torch.save(model.state_dict(), f'test_{epoch+1}.pth')
