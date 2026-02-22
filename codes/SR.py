import os
import random
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

# ===============================
# MATPLOTLIB FIX (Windows Live Plot)
# ===============================
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
SCALE = 4
PATCH_SIZE = 96
BATCH_SIZE = 4
EPOCHS = 300
LR_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os

import os

# -------------------- Absolute base paths --------------------
BASE_TRAIN_DIR = r"C:\Users\inder\Documents\AI_Image_Restoration\data\RealSR(V3)"
BASE_VAL_DIR = r"C:\Users\inder\Documents\MTechProject\Project\images\RealSR(V3)"

CAMERAS = ["Canon", "Nikon"]  # cameras to include

# -------------------- Helper function --------------------
def get_subfolders(base_path, camera, split="Train"):
    """
    Returns a list of all subfolders inside a camera's Train/Test folder.
    If the folder doesn't exist, it skips it instead of crashing.
    """
    path = os.path.join(base_path, camera, split)
    if not os.path.exists(path):
        print(f"Warning: Folder not found, skipping: {path}")
        return []
    return [os.path.join(path, f) for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))]

# -------------------- Create Train and Validation lists --------------------
TRAIN_DIRS = []
VAL_DIRS = []

for cam in CAMERAS:
    TRAIN_DIRS += get_subfolders(BASE_TRAIN_DIR, cam, split="Train")
    VAL_DIRS += get_subfolders(BASE_VAL_DIR, cam, split="Test")

# -------------------- Print to verify --------------------
print("Train folders found:")
for d in TRAIN_DIRS:
    print(d)

print("\nValidation folders found:")
for d in VAL_DIRS:
    print(d)

# ===============================
# DATASET WITH PATCH TRAINING
# ===============================
class RealSRDataset(Dataset):
    def __init__(self, folders, train=True):
        self.train = train
        self.lr_images = []
        self.hr_images = []
        self.transform = transforms.ToTensor()

        for folder in folders:
            for file in os.listdir(folder):
                if "_LR4" in file:
                    lr_path = os.path.join(folder, file)
                    hr_path = os.path.join(folder, file.replace("_LR4", "_HR"))

                    if os.path.exists(hr_path):
                        self.lr_images.append(lr_path)
                        self.hr_images.append(hr_path)

        print(f"Loaded {len(self.lr_images)} image pairs.")

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):

        lr = Image.open(self.lr_images[idx]).convert("RGB")
        hr = Image.open(self.hr_images[idx]).convert("RGB")

        if self.train:
            lr_w, lr_h = lr.size

            x = random.randint(0, lr_w - PATCH_SIZE)
            y = random.randint(0, lr_h - PATCH_SIZE)

            lr = lr.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            hr = hr.crop((
                x * SCALE,
                y * SCALE,
                (x + PATCH_SIZE) * SCALE,
                (y + PATCH_SIZE) * SCALE
            ))

            if random.random() > 0.5:
                lr = TF.hflip(lr)
                hr = TF.hflip(hr)

            if random.random() > 0.5:
                lr = TF.vflip(lr)
                hr = TF.vflip(hr)

        lr = self.transform(lr)
        hr = self.transform(hr)

        return lr, hr

# ===============================
# METRICS
# ===============================
def calculate_psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    return 10 * torch.log10(1 / mse)

def calculate_ssim(sr, hr):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(sr, 3, 1, 1)
    mu_y = F.avg_pool2d(hr, 3, 1, 1)

    sigma_x = F.avg_pool2d(sr * sr, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(hr * hr, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(sr * hr, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return ssim_map.mean()

# ===============================
# CHARBONNIER LOSS
# ===============================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))

# ===============================
# WINDOW FUNCTIONS
# ===============================
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size,
               W // window_size, window_size)
    windows = x.permute(0,2,4,3,5,1).contiguous()
    return windows.view(-1, window_size*window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H*W/window_size/window_size))
    x = windows.view(B, H//window_size, W//window_size,
                     window_size, window_size, -1)
    x = x.permute(0,5,1,3,2,4).contiguous()
    return x.view(B, -1, H, W)

# ===============================
# SWIN BLOCK
# ===============================
class SwinBlock(nn.Module):
    def __init__(self, dim, window_size=8, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        if self.shift:
            x = torch.roll(x, shifts=(-ws//2, -ws//2), dims=(2,3))

        windows = window_partition(x, ws)
        windows = self.norm1(windows)

        attn_out,_ = self.attn(windows, windows, windows)
        windows = windows + attn_out
        windows = windows + self.mlp(self.norm2(windows))

        x = window_reverse(windows, ws, Hp, Wp)

        if self.shift:
            x = torch.roll(x, shifts=(ws//2, ws//2), dims=(2,3))

        return x[:, :, :H, :W]

# ===============================
# MODEL (DEEPER VERSION)
# ===============================
class MiniSwinIR(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.conv_first = nn.Conv2d(3, dim, 3, 1, 1)

        self.blocks = nn.Sequential(
            SwinBlock(dim, shift=False),
            SwinBlock(dim, shift=True),
            SwinBlock(dim, shift=False),
            SwinBlock(dim, shift=True),
        )

        self.conv_mid = nn.Conv2d(dim, dim, 3,1,1)

        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim*(SCALE**2),3,1,1),
            nn.PixelShuffle(SCALE),
            nn.Conv2d(dim,3,3,1,1)
        )

    def forward(self,x):
        x = self.conv_first(x)
        res = x
        x = self.blocks(x)
        x = self.conv_mid(x)
        x = x + res
        return self.upsample(x)

# ===============================
# LOADERS
# ===============================
train_dataset = RealSRDataset(TRAIN_DIRS, train=True)
val_dataset = RealSRDataset(VAL_DIRS, train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ===============================
# TRAINING
# ===============================
model = MiniSwinIR().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = CharbonnierLoss()

best_psnr = 0

plt.ion()
fig, axs = plt.subplots(2,2, figsize=(10,8))

train_loss_list = []
train_psnr_list = []
val_psnr_list = []
val_ssim_list = []

# ===============================
# LOG FILE SETUP
# ===============================
log_file = "training_log.txt"

with open(log_file, "w") as f:
    f.write("Epoch,Train_Loss,Train_PSNR,Val_PSNR,Val_SSIM,Learning_Rate\n")


for epoch in range(EPOCHS):

    model.train()
    total_loss = 0
    train_psnr = 0

    for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        lr, hr = lr.to(DEVICE), hr.to(DEVICE)

        sr = model(lr)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_psnr += calculate_psnr(sr, hr).item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    train_psnr /= len(train_loader)

    model.eval()
    val_psnr = 0
    val_ssim = 0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            sr = model(lr)

            val_psnr += calculate_psnr(sr, hr).item()
            val_ssim += calculate_ssim(sr, hr).item()

    val_psnr /= len(val_loader)
    val_ssim /= len(val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss : {avg_loss:.6f}")
    print(f"Train PSNR : {train_psnr:.4f}")
    print(f"Val PSNR   : {val_psnr:.4f}")
    print(f"Val SSIM   : {val_ssim:.4f}")
    
    current_lr = optimizer.param_groups[0]['lr']

    with open(log_file, "a") as f:
     f.write(f"{epoch+1},{avg_loss:.6f},{train_psnr:.4f},{val_psnr:.4f},{val_ssim:.4f},{current_lr:.8f}\n")


    train_loss_list.append(avg_loss)
    train_psnr_list.append(train_psnr)
    val_psnr_list.append(val_psnr)
    val_ssim_list.append(val_ssim)

    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save(model.state_dict(), "best_mini_swinir.pth")
        print("Best model saved!")

    axs[0,0].clear(); axs[0,0].plot(train_loss_list); axs[0,0].set_title("Train Loss")
    axs[0,1].clear(); axs[0,1].plot(train_psnr_list); axs[0,1].set_title("Train PSNR")
    axs[1,0].clear(); axs[1,0].plot(val_psnr_list); axs[1,0].set_title("Val PSNR")
    axs[1,1].clear(); axs[1,1].plot(val_ssim_list); axs[1,1].set_title("Val SSIM")

    plt.tight_layout()
    plt.pause(0.01)    
    plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')


plt.ioff()
plt.show()
