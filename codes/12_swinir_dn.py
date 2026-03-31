import os
import random
import csv
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
# MATPLOTLIB (LIVE PLOT FIX)
# ===============================
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
PATCH_SIZE = 64
BATCH_SIZE = 1
EPOCHS = 300
LR_RATE = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

BASE_TRAIN_DIR = r"C:\Users\inder\Documents\AI_Image_Restoration\data\RealSR(V3)"
BASE_VAL_DIR   = r"C:\Users\inder\Documents\MTechProject\Project\images\RealSR(V3)"
CAMERAS = ["Canon", "Nikon"]

# ===============================
# DATA LOADER
# ===============================
def get_subfolders(base_path, camera, split="Train"):
    path = os.path.join(base_path, camera, split)
    if not os.path.exists(path):
        print("Missing path:", path)
        return []
    return [os.path.join(path, f) for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))]

TRAIN_DIRS, VAL_DIRS = [], []
for cam in CAMERAS:
    TRAIN_DIRS += get_subfolders(BASE_TRAIN_DIR, cam, "Train")
    VAL_DIRS   += get_subfolders(BASE_VAL_DIR, cam, "Test")

# ===============================
# DATASET
# ===============================
class DenoiseDataset(Dataset):
    def __init__(self, folders, train=True):
        self.train = train
        self.images = []
        self.transform = transforms.ToTensor()

        for folder in folders:
            for file in os.listdir(folder):
                if "_HR" in file:
                    self.images.append(os.path.join(folder, file))

        print(f"[Dataset] Loaded {len(self.images)} images. Train={self.train}")

    def add_noise(self, img):
        std = random.uniform(5, 25) / 255.0
        gaussian = torch.randn_like(img) * std
        poisson = torch.poisson(img * 255.0) / 255.0 - img
        noisy = img + gaussian + 0.5 * poisson
        return torch.clamp(noisy, 0.0, 1.0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")

        if self.train:
            w, h = img.size
            x = random.randint(0, w - PATCH_SIZE)
            y = random.randint(0, h - PATCH_SIZE)
            img = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

        clean = self.transform(img)
        noisy = self.add_noise(clean)

        return noisy, clean

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

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return ssim.mean()

# ===============================
# LOSS
# ===============================
class CharbonnierLoss(nn.Module):
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + 1e-6))

# ===============================
# SWIN BLOCK
# ===============================
def window_partition(x, ws):
    B,C,H,W = x.shape
    x = x.view(B, C, H//ws, ws, W//ws, ws)
    return x.permute(0,2,4,3,5,1).contiguous().view(-1, ws*ws, C)

def window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H*W/ws/ws))
    x = windows.view(B, H//ws, W//ws, ws, ws, -1)
    return x.permute(0,5,1,3,2,4).contiguous().view(B, -1, H, W)

class SwinBlock(nn.Module):
    def __init__(self, dim, shift=False):
        super().__init__()
        self.ws = 8
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 2, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )

    def forward(self, x):
        B,C,H,W = x.shape
        ws = self.ws

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0,pad_w,0,pad_h))
        _,_,Hp,Wp = x.shape

        if self.shift:
            x = torch.roll(x, (-ws//2,-ws//2),(2,3))

        windows = window_partition(x, ws)
        windows = self.norm1(windows)

        attn,_ = self.attn(windows, windows, windows)
        windows = windows + attn
        windows = windows + self.mlp(self.norm2(windows))

        x = window_reverse(windows, ws, Hp, Wp)

        if self.shift:
            x = torch.roll(x, (ws//2,ws//2),(2,3))

        return x[:,:,:H,:W]

# ===============================
# MODEL
# ===============================
class SwinIR_DN(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.conv_first = nn.Conv2d(3, dim, 3,1,1)

        self.blocks = nn.Sequential(
            SwinBlock(dim, False),
            SwinBlock(dim, True),
            SwinBlock(dim, False),
            SwinBlock(dim, True),
        )

        self.conv_mid = nn.Conv2d(dim, dim, 3,1,1)
        self.conv_last = nn.Conv2d(dim, 3, 3,1,1)

    def forward(self, x):
        x = self.conv_first(x)
        res = x
        x = self.blocks(x)
        x = self.conv_mid(x)
        x = x + res
        return self.conv_last(x)

# ===============================
# TILE INFERENCE
# ===============================
def tile_inference(model, img, tile_size=128, overlap=8):

    device = img.device
    B, C, H, W = img.shape
    stride = tile_size - overlap

    output = torch.zeros_like(img, device=device)
    weight = torch.zeros_like(img, device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            y1 = y
            x1 = x
            y2 = min(y1 + tile_size, H)
            x2 = min(x1 + tile_size, W)

            y1 = max(y2 - tile_size, 0)
            x1 = max(x2 - tile_size, 0)

            tile = img[:, :, y1:y2, x1:x2]

            pred_noise = model(tile)
            tile_out = tile - pred_noise

            output[:, :, y1:y2, x1:x2] += tile_out
            weight[:, :, y1:y2, x1:x2] += 1

    return output / weight

# ===============================
# LOADERS
# ===============================
train_loader = DataLoader(DenoiseDataset(TRAIN_DIRS, True),
                          batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(DenoiseDataset(VAL_DIRS, False),
                        batch_size=1)

# ===============================
# TRAIN SETUP
# ===============================
model = SwinIR_DN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = CharbonnierLoss()

best_psnr = 0

# ===============================
# LIVE PLOT SETUP
# ===============================
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

train_loss_list = []
train_psnr_list = []
val_psnr_list = []
val_ssim_list = []

# ===============================
# TRAIN LOOP
# ===============================
for epoch in range(EPOCHS):

    print(f"\n===== EPOCH {epoch+1} =====")

    model.train()
    total_loss = 0
    train_psnr = 0

    for noisy, clean in tqdm(train_loader):

        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        pred_noise = model(noisy)
        loss = criterion(pred_noise, noisy - clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        denoised = noisy - pred_noise

        total_loss += loss.item()
        train_psnr += calculate_psnr(denoised, clean).item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    train_psnr /= len(train_loader)

    train_loss_list.append(avg_loss)
    train_psnr_list.append(train_psnr)

    # ================= VALIDATION
    print("Validation (10 images only)...")

    model.eval()
    val_psnr, val_ssim = 0, 0
    count = 0

    with torch.no_grad():
        for i, (noisy, clean) in enumerate(val_loader):

            if i >= 10:
                break

            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            denoised = tile_inference(model, noisy)

            val_psnr += calculate_psnr(denoised, clean).item()
            val_ssim += calculate_ssim(denoised, clean).item()
            count += 1

    val_psnr /= count
    val_ssim /= count

    val_psnr_list.append(val_psnr)
    val_ssim_list.append(val_ssim)

    print(f"PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f}")

    # ================= SAVE MODEL
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save(model.state_dict(), "best_swinir_dn.pth")
        print("Saved best model!")

    # ================= LIVE PLOTS
    axs[0,0].clear()
    axs[0,0].plot(train_loss_list)
    axs[0,0].set_title("Train Loss")

    axs[0,1].clear()
    axs[0,1].plot(train_psnr_list)
    axs[0,1].set_title("Train PSNR")

    axs[1,0].clear()
    axs[1,0].plot(val_psnr_list)
    axs[1,0].set_title("Val PSNR")

    axs[1,1].clear()
    axs[1,1].plot(val_ssim_list)
    axs[1,1].set_title("Val SSIM")

    plt.tight_layout()
    plt.pause(0.01)
    plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')

plt.ioff()
plt.show()

print("Training done!")
