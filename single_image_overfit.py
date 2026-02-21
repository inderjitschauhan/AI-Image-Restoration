from pathlib import Path
import random
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ---------------------------
# Random Image Selection
# ---------------------------

def get_random_image(data_dir="data"):
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path.resolve()}")

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    image_paths = [
        p for p in data_path.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        raise ValueError(f"No images found inside {data_path.resolve()}")

    return random.choice(image_paths)


selected_image = get_random_image("data")
IMAGE_PATH = str(selected_image)

print("Selected image name :", selected_image.name)
print("Selected folder     :", selected_image.parent)
print("Selected image path :", selected_image.resolve())


# ---------------------------
# Configuration
# ---------------------------

PATCH_SIZE = 64
SIGMA = 25
EPOCHS = 2000
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# PSNR Function
# ---------------------------

def psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    return 10 * torch.log10(1.0 / mse)


# ---------------------------
# DnCNN Model
# ---------------------------

class DnCNN(nn.Module):
    def __init__(self, depth=8, channels=32):
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
        return self.net(x)


# ---------------------------
# Load Image
# ---------------------------

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Failed to load image: {IMAGE_PATH}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0

H, W, _ = img.shape
y = np.random.randint(0, H - PATCH_SIZE)
x = np.random.randint(0, W - PATCH_SIZE)

clean_np = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]


# ---------------------------
# Add Noise
# ---------------------------

noise_np = np.random.normal(0, SIGMA / 255.0, clean_np.shape).astype(np.float32)
noisy_np = np.clip(clean_np + noise_np, 0, 1)

val_noise_np = np.random.normal(0, SIGMA / 255.0, clean_np.shape).astype(np.float32)
val_noisy_np = np.clip(clean_np + val_noise_np, 0, 1)


# ---------------------------
# Convert to Tensors
# ---------------------------

clean = torch.from_numpy(clean_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
noisy = torch.from_numpy(noisy_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
noise = torch.from_numpy(noise_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

val_noisy = torch.from_numpy(val_noisy_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
val_noise = torch.from_numpy(val_noise_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


# ---------------------------
# Model Setup
# ---------------------------

model = DnCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ---------------------------
# Training Loop
# ---------------------------

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

train_losses = []
val_losses = []
train_psnr = []
val_psnr = []

print("\nOverfitting on a single image\n")

for epoch in range(1, EPOCHS + 1):

    model.train()
    optimizer.zero_grad()

    pred_noise = model(noisy)
    loss = criterion(pred_noise, noise)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred_noise = model(val_noisy)
        val_loss = criterion(val_pred_noise, val_noise)

        denoised = torch.clamp(noisy - pred_noise, 0, 1)
        val_denoised = torch.clamp(val_noisy - val_pred_noise, 0, 1)

        p_train = psnr(denoised, clean)
        p_val = psnr(val_denoised, clean)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_psnr.append(p_train.item())
    val_psnr.append(p_val.item())

    if epoch % 100 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:4d} | "
            f"Train Loss: {loss.item():.3e} | "
            f"Val Loss: {val_loss.item():.3e} | "
            f"Train PSNR: {p_train.item():.2f} dB | "
            f"Val PSNR: {p_val.item():.2f} dB"
        )

    if epoch % 10 == 0:
        ax1.clear()
        ax1.plot(train_losses, label="Train Loss")
        ax1.plot(val_losses, label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.clear()
        ax2.plot(train_psnr, label="Train PSNR")
        ax2.plot(val_psnr, label="Val PSNR")
        ax2.set_title("PSNR")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.tight_layout()
        plt.pause(0.001)

plt.ioff()
plt.show()

print("\nOverfit test completed")