import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ===============================
# CONFIG
# ===============================
SET5_DIR = "data/set5"
DNCNN_PATH = "weights/dncnn/dncnn_trained_best50.pth"
SR_PATH = "weights/SR/best_mini_swinir.pth"

NOISE_STD = 25
SCALE = 4
DEVICE = torch.device("cpu")   # Change to "cuda" if GPU stable

# ===============================
# DnCNN (8 layers, 32 channels)
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
# SWIN BLOCK
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
# MINI SWINIR
# ===============================
class MiniSwinIR(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.conv_first = nn.Conv2d(3, dim, 3,1,1)

        self.blocks = nn.Sequential(
            SwinBlock(dim, shift=False),
            SwinBlock(dim, shift=True),
            SwinBlock(dim, shift=False),
            SwinBlock(dim, shift=True),
        )

        self.conv_mid = nn.Conv2d(dim, dim,3,1,1)

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
# LOAD MODELS
# ===============================
print("Loading models...")

dncnn = DnCNN().to(DEVICE)
dncnn.load_state_dict(torch.load(DNCNN_PATH, map_location=DEVICE))
dncnn.eval()

swinir = MiniSwinIR().to(DEVICE)
swinir.load_state_dict(torch.load(SR_PATH, map_location=DEVICE))
swinir.eval()

print("Models loaded successfully!")

# ===============================
# LOAD RANDOM IMAGE
# ===============================
images = [f for f in os.listdir(SET5_DIR)
          if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]

img_name = random.choice(images)
img_path = os.path.join(SET5_DIR, img_name)

print("Selected Image:", img_name)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0

# ===============================
# ADD NOISE
# ===============================
noise = np.random.normal(0, NOISE_STD/255.0, img.shape).astype(np.float32)
noisy = np.clip(img + noise, 0, 1)

noisy_tensor = torch.from_numpy(noisy).permute(2,0,1).unsqueeze(0).to(DEVICE)

# ===============================
# DENOISING
# ===============================
with torch.no_grad():
    predicted_noise = dncnn(noisy_tensor)
    denoised_tensor = noisy_tensor - predicted_noise

denoised_tensor = denoised_tensor.clamp(0,1)

# ===============================
# SUPER RESOLUTION
# ===============================
with torch.no_grad():
    sr_tensor = swinir(denoised_tensor)

sr_tensor = sr_tensor.clamp(0,1)

# ===============================
# METRICS
# ===============================
clean_np = img
noisy_np = noisy
denoised_np = denoised_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
sr_np = sr_tensor.squeeze(0).permute(1,2,0).cpu().numpy()

h, w, _ = clean_np.shape
clean_up = cv2.resize(clean_np, (w*SCALE, h*SCALE), interpolation=cv2.INTER_CUBIC)

psnr_noisy = peak_signal_noise_ratio(clean_np, noisy_np, data_range=1.0)
psnr_denoised = peak_signal_noise_ratio(clean_np, denoised_np, data_range=1.0)
psnr_sr = peak_signal_noise_ratio(clean_up, sr_np, data_range=1.0)

ssim_noisy = structural_similarity(clean_np, noisy_np, channel_axis=2, data_range=1.0)
ssim_denoised = structural_similarity(clean_np, denoised_np, channel_axis=2, data_range=1.0)
ssim_sr = structural_similarity(clean_up, sr_np, channel_axis=2, data_range=1.0)

print("\n==============================")
print("📊 Image Quality Metrics")
print("==============================")
print(f"PSNR (Noisy vs Clean):      {psnr_noisy:.2f} dB")
print(f"PSNR (Denoised vs Clean):   {psnr_denoised:.2f} dB")
print(f"PSNR (SR vs Clean Up):      {psnr_sr:.2f} dB")
print()
print(f"SSIM (Noisy vs Clean):      {ssim_noisy:.4f}")
print(f"SSIM (Denoised vs Clean):   {ssim_denoised:.4f}")
print(f"SSIM (SR vs Clean Up):      {ssim_sr:.4f}")
print("==============================\n")

# ===============================
# DISPLAY
# ===============================
input_image = Image.fromarray((img*255).astype(np.uint8))
final_image = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Original Clean Input")
plt.imshow(input_image)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Denoised + 4x SR Output")
plt.imshow(final_image)
plt.axis("off")

plt.tight_layout()
plt.show()


import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ====================================
# CONFIG
# ====================================
DATA_DIR = r"C:\Users\inder\Documents\AI_Image_Restoration\data\RealSR(V3)\Canon\Train\4"
SCALE = 4
DEVICE = torch.device("cpu")

results = []

# Get LR4 images
image_list = [f for f in os.listdir(DATA_DIR) if "_LR4" in f]

# Random 10 images
random_images = random.sample(image_list, min(10, len(image_list)))

print("Selected Images:")
for name in random_images:
    print(" -", name)

print("\nProcessing...\n")

for img_name in random_images:

    lr_path = os.path.join(DATA_DIR, img_name)
    hr_path = lr_path.replace("_LR4", "_HR")

    if not os.path.exists(hr_path):
        print("HR not found:", img_name)
        continue

    # ===============================
    # LOAD LR
    # ===============================
    lr = cv2.imread(lr_path)
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    lr = lr.astype(np.float32) / 255.0

    lr_tensor = torch.from_numpy(lr).permute(2,0,1).unsqueeze(0).to(DEVICE)

    # ===============================
    # SWINIR (DIRECT)
    # ===============================
    with torch.no_grad():
        sr_tensor = swinir(lr_tensor)

    sr_tensor = sr_tensor.clamp(0,1)
    sr = sr_tensor.squeeze(0).permute(1,2,0).cpu().numpy()

    # ===============================
    # LOAD HR
    # ===============================
    hr = cv2.imread(hr_path)
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    hr = hr.astype(np.float32) / 255.0

    # ===============================
    # BICUBIC (LR → HR)
    # ===============================
    h_lr, w_lr, _ = lr.shape
    bicubic = cv2.resize(
        lr,
        (w_lr*SCALE, h_lr*SCALE),
        interpolation=cv2.INTER_CUBIC
    )

    # ===============================
    # METRICS
    # ===============================
    psnr_bic = peak_signal_noise_ratio(hr, bicubic, data_range=1.0)
    ssim_bic = structural_similarity(
        hr, bicubic,
        channel_axis=2,
        data_range=1.0
    )

    psnr_sr = peak_signal_noise_ratio(hr, sr, data_range=1.0)
    ssim_sr = structural_similarity(
        hr, sr,
        channel_axis=2,
        data_range=1.0
    )

    results.append([
        img_name,
        psnr_bic, ssim_bic,
        psnr_sr, ssim_sr
    ])

    print(f"{img_name} done.")

# ====================================
# CREATE TABLE
# ====================================
columns = [
    "Image",
    "PSNR_Bicubic", "SSIM_Bicubic",
    "PSNR_SwinIR", "SSIM_SwinIR"
]

df = pd.DataFrame(results, columns=columns)

# Add average row
avg_values = df.iloc[:,1:].mean()
df.loc[len(df)] = ["Average"] + list(avg_values)

print("\n==============================")
print(df)
print("==============================")

df.to_csv("RealSR_SwinIR_only_random10.csv", index=False)
print("\nSaved as RealSR_SwinIR_only_random10.csv")