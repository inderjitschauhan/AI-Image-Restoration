import os
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ================= CONFIG =================
DATA_DIRS = [
    os.path.abspath(r"./data/RealSR(V3)/Canon/Train/4"),
    os.path.abspath(r"./data/RealSR(V3)/Nikon/Train/4")
]

SAVE_DIR = os.path.abspath(r"./weights/dncnn")
os.makedirs(SAVE_DIR, exist_ok=True)

PATCH_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-4
PATCHES_PER_IMAGE = 4
SIGMA = 15   # Read noise level
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

best_psnr = 0

# ================= PSNR =================
def psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return 10 * torch.log10(1.0 / mse)

# ================= DATASET =================
class RealNoiseDataset(Dataset):
    def __init__(self, roots, patch_size=128, patches_per_image=1):
        if isinstance(roots, str):
            roots = [roots]

        self.hr_images = []
        for root in roots:
            hr_list = sorted(glob.glob(os.path.join(root, "*_HR.*")))
            if len(hr_list) == 0:
                raise ValueError(f"No HR images found in {root}")
            self.hr_images.extend(hr_list)

        print(f"Total HR images: {len(self.hr_images)}")

        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.indices = []
        for i in range(len(self.hr_images)):
            for _ in range(self.patches_per_image):
                self.indices.append(i)

        print(f"Total patches: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        hr = Image.open(self.hr_images[real_idx]).convert("RGB")

        # Random patch
        w, h = hr.size
        pw = min(self.patch_size, w)
        ph = min(self.patch_size, h)
        x = random.randint(0, w - pw)
        y = random.randint(0, h - ph)
        hr = hr.crop((x, y, x + pw, y + ph))

        # Data augmentation
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            hr = hr.rotate(angle)

        # To tensor
        hr = torch.from_numpy(np.array(hr)).permute(2, 0, 1).float() / 255.0
        img = hr.clone()

        # ---------------- REALISTIC NOISE ----------------
        # Shot noise (Poisson, signal dependent)
        shot_noise = torch.poisson(img * 255.0) / 255.0 - img23456798 

        # Read noise (Gaussian)
        read_noise = torch.randn_like(img) * (SIGMA / 255.0)

        noisy = img + shot_noise + read_noise 
        noisy = torch.clamp(noisy, 0.0, 1.0)

        return noisy, hr

# ================= MODEL =================
class DnCNN(nn.Module):
    def __init__(self, depth=17, channels=64):
        super().__init__()
        layers = [nn.Conv2d(3, channels, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ]
        layers += [nn.Conv2d(channels, 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        return x - noise  # Residual learning

# ================= MAIN TRAINING =================
if __name__ == "__main__":

    dataset = RealNoiseDataset(DATA_DIRS, PATCH_SIZE, PATCHES_PER_IMAGE)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = DnCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loss_hist, val_loss_hist, val_psnr_hist = [], [], []

    print("Training started on device:", DEVICE)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for epoch in range(1, EPOCHS + 1):

        # -------- TRAIN --------
        model.train()
        train_loss = 0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        val_psnr_val = 0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

                pred = model(noisy)
                pred = torch.clamp(pred, 0.0, 1.0)

                loss = criterion(pred, clean)
                val_loss += loss.item()
                val_psnr_val += psnr(pred, clean).item()

        val_loss /= len(val_loader)
        val_psnr_val /= len(val_loader)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_psnr_hist.append(val_psnr_val)

        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.5f} | "
              f"Val Loss: {val_loss:.5f} | "
              f"Val PSNR: {val_psnr_val:.2f} dB")

        # -------- LIVE PLOT --------
        ax1.clear()
        ax1.plot(train_loss_hist, label="Train Loss")
        ax1.plot(val_loss_hist, label="Val Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.clear()
        ax2.plot(val_psnr_hist, label="Val PSNR")
        ax2.set_title("Validation PSNR")
        ax2.legend()
        ax2.grid(True)

        plt.pause(0.1)

        # -------- SAVE --------
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "dncnn_real_last_lrsr.pth"))

        if val_psnr_val > best_psnr:
            best_psnr = val_psnr_val
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "dncnn_real__lrsr.pth"))
            print(f"Best model saved. PSNR = {val_psnr_val:.2f} dB")

    plt.ioff()
    plt.show()

    print("Training finished!")
