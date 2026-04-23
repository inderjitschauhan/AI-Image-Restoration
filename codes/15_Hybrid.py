import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
PATCH_SIZE = 64
BATCH_SIZE = 4
EPOCHS = 1
LR_RATE = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

BASE_TRAIN_DIR = r"C:\Users\inder\Documents\AI_Image_Restoration\data\RealSR(V3)"
BASE_VAL_DIR   = r"C:\Users\inder\Documents\MTechProject\Project\images\RealSR(V3)"
CAMERAS = ["Canon", "Nikon"]

# ===============================
# DATASET
# ===============================
def get_subfolders(base_path, camera, split):
    path = os.path.join(base_path, camera, split)
    return [os.path.join(path, f) for f in os.listdir(path)]

class DenoiseDataset(Dataset):
    def __init__(self, folders, train=True):
        self.train = train
        self.images = []
        self.transform = transforms.ToTensor()

        for folder in folders:
            for file in os.listdir(folder):
                if "_HR" in file:
                    self.images.append(os.path.join(folder, file))

        print(f"Loaded {len(self.images)} images | Train={train}")

    def add_noise(self, img):
        std = random.uniform(5, 25) / 255.0
        gaussian = torch.randn_like(img) * std
        poisson = torch.poisson(img * 255.0) / 255.0 - img
        return torch.clamp(img + gaussian + 0.5 * poisson, 0, 1)

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
def psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    return 10 * torch.log10(1 / mse)

def ssim(sr, hr):
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(sr, 3, 1, 1)
    mu_y = F.avg_pool2d(hr, 3, 1, 1)

    sigma_x = F.avg_pool2d(sr*sr, 3, 1, 1) - mu_x**2
    sigma_y = F.avg_pool2d(hr*hr, 3, 1, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(sr*hr, 3, 1, 1) - mu_x*mu_y

    return (((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) /
            ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))).mean()

# ===============================
# LOSS
# ===============================
class CharbonnierLoss(nn.Module):
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + 1e-6))

# ===============================
# SWIN BLOCK
# ===============================
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

        if self.shift:
            x = torch.roll(x, (-ws//2,-ws//2),(2,3))

        x = x.flatten(2).transpose(1,2)
        x = self.norm1(x)
        attn,_ = self.attn(x,x,x)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        x = x.transpose(1,2).view(B,C,H,W)

        if self.shift:
            x = torch.roll(x, (ws//2,ws//2),(2,3))

        return x

# ===============================
# HYBRID MODEL
# ===============================
class HybridDenoiser(nn.Module):
    def __init__(self, dim=48):
        super().__init__()

        self.input_conv = nn.Conv2d(3, dim, 3,1,1)

        self.cnn_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3,1,1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3,1,1),
            nn.ReLU()
        )

        self.trans_branch = nn.Sequential(
            SwinBlock(dim),
            SwinBlock(dim, True),
            SwinBlock(dim)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3,1,1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3,1,1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3,1,1)
        )

        self.output_conv = nn.Conv2d(dim, 3, 3,1,1)

    def forward(self, x):
        x0 = self.input_conv(x)

        cnn_feat = self.cnn_branch(x0)
        trans_feat = self.trans_branch(x0)

        fused = torch.cat([cnn_feat, trans_feat], dim=1)
        fused = self.fusion(fused)

        refined = self.refine(fused)

        return self.output_conv(refined)

# ===============================
# TILE INFERENCE
# ===============================
def tile_inference(model, img):
    return img - model(img)

# ===============================
# LOADERS
# ===============================
train_dirs = []
val_dirs = []

for cam in CAMERAS:
    train_dirs += get_subfolders(BASE_TRAIN_DIR, cam, "Train")
    val_dirs   += get_subfolders(BASE_VAL_DIR, cam, "Test")

train_loader = DataLoader(DenoiseDataset(train_dirs, True),
                          batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(DenoiseDataset(val_dirs, False),
                        batch_size=1)

# ===============================
# TRAIN SETUP
# ===============================
model = HybridDenoiser().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = CharbonnierLoss()

best_psnr = 0

# ===============================
# TRAIN LOOP
# ===============================
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for noisy, clean in tqdm(train_loader):

        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        pred_noise = model(noisy)
        loss = criterion(pred_noise, noisy - clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"\nEpoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    # ===== VALIDATION =====
    model.eval()
    v_psnr, v_ssim = 0, 0

    with torch.no_grad():
        for i, (noisy, clean) in enumerate(val_loader):
            if i >= 10:
                break

            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            out = tile_inference(model, noisy)

            v_psnr += psnr(out, clean).item()
            v_ssim += ssim(out, clean).item()

    v_psnr /= 10
    v_ssim /= 10

    print(f"Val PSNR: {v_psnr:.3f} | SSIM: {v_ssim:.4f}")

    if v_psnr > best_psnr:
        best_psnr = v_psnr
        torch.save(model.state_dict(), "best_hybrid.pth")
        print("Saved best model!")

print("Training Complete!")
