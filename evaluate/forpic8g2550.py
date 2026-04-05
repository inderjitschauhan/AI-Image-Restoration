import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGMA = 50                 # 25 or 50
NOISE_TYPE = "gaussian"    # gaussian / real

MODEL_PATH = "weights/dncnn/Dncnn_8_Gaussian_Best_50.pth"

# ===============================
# YOUR MODEL (MATCH TRAINING)
# ===============================
class DnCNN(nn.Module):
    def __init__(self, depth=8, channels=32):
        super().__init__()
        layers = [
            nn.Conv2d(3, channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.ReLU(inplace=True)
            ]
        layers += [nn.Conv2d(channels, 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ===============================
# METRICS
# ===============================
def calculate_psnr(x, y):
    mse = F.mse_loss(x, y)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    sigma1 = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

    return (((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) /
           ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))).mean()

# ===============================
# NOISE
# ===============================
def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255)

def add_real_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    poisson = np.random.poisson(img) - img
    return np.clip(img + noise + 0.3 * poisson, 0, 255)

# ===============================
# LOAD IMAGE
# ===============================
Tk().withdraw()
path = askopenfilename(title="Select Image")

img = Image.open(path).convert("RGB")
img_np = np.array(img).astype(np.float32)

# ===============================
# ADD NOISE
# ===============================
if NOISE_TYPE == "gaussian":
    noisy_np = add_gaussian_noise(img_np, SIGMA)
else:
    noisy_np = add_real_noise(img_np, SIGMA)

noisy_np = noisy_np.astype(np.uint8)
noisy_img = Image.fromarray(noisy_np)

# ===============================
# LOAD MODEL
# ===============================
model = DnCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===============================
# INFERENCE
# ===============================
transform = transforms.ToTensor()

noisy_tensor = transform(noisy_img).unsqueeze(0).to(DEVICE)
clean_tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred_noise = model(noisy_tensor)
    denoised = torch.clamp(noisy_tensor - pred_noise, 0, 1)

# ===============================
# METRICS
# ===============================
psnr_noisy = calculate_psnr(noisy_tensor, clean_tensor)
psnr_denoised = calculate_psnr(denoised, clean_tensor)

ssim_noisy = calculate_ssim(noisy_tensor, clean_tensor)
ssim_denoised = calculate_ssim(denoised, clean_tensor)

# ===============================
# CONVERT
# ===============================
den_np = denoised.squeeze().cpu().numpy().transpose(1,2,0)
den_np = (den_np * 255).astype(np.uint8)
den_img = Image.fromarray(den_np)

# ===============================
# DISPLAY
# ===============================
def put_text(ax, text, y=0):
    ax.text(
        0.02, 0.95 - y,
        text,
        transform=ax.transAxes,
        fontsize=10,
        color='yellow',
        verticalalignment='top',
        bbox=dict(facecolor='black', alpha=0.6)
    )

plt.figure(figsize=(15,5))

# Original
ax1 = plt.subplot(1,3,1)
ax1.imshow(img)
ax1.set_title("Original")
ax1.axis("off")

# Noisy
ax2 = plt.subplot(1,3,2)
ax2.imshow(noisy_img)
ax2.set_title("Noisy")
ax2.axis("off")
put_text(ax2, f"{NOISE_TYPE} σ={SIGMA}")
put_text(ax2, f"PSNR: {psnr_noisy:.2f}", 0.08)
put_text(ax2, f"SSIM: {ssim_noisy:.4f}", 0.16)

# Denoised
ax3 = plt.subplot(1,3,3)
ax3.imshow(den_img)
ax3.set_title("DnCNN Output")
ax3.axis("off")
put_text(ax3, f"PSNR: {psnr_denoised:.2f}")
put_text(ax3, f"SSIM: {ssim_denoised:.4f}", 0.08)

plt.tight_layout()
plt.show()
