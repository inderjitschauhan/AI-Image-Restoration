import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
import cv2
import numpy as np

# ===============================
# CONFIG
# ===============================
SCALE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "weights/SR/best_swinir_sr.pth"

# ===============================
# METRICS
# ===============================
def calculate_psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    sigma1 = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_map.mean()

# ===============================
# WINDOW FUNCTIONS
# ===============================
def window_partition(x, ws):
    B, C, H, W = x.shape
    x = x.view(B, C, H // ws, ws, W // ws, ws)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return windows.view(-1, ws * ws, C)

def window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, -1, H, W)

# ===============================
# SWIN BLOCK
# ===============================
class SwinBlock(nn.Module):
    def __init__(self, dim, window_size=8, shift=False):
        super().__init__()
        self.ws = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ws = min(self.ws, H, W)
        if ws < 1:
            ws = 1

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = x.shape

        if self.shift:
            x = torch.roll(x, (-ws // 2, -ws // 2), (2, 3))

        windows = window_partition(x, ws)
        windows = self.norm1(windows)

        attn_out, _ = self.attn(windows, windows, windows)
        windows = windows + attn_out
        windows = windows + self.mlp(self.norm2(windows))

        x = window_reverse(windows, ws, Hp, Wp)

        if self.shift:
            x = torch.roll(x, (ws // 2, ws // 2), (2, 3))

        return x[:, :, :H, :W]

# ===============================
# MINI SWINIR MODEL
# ===============================
class MiniSwinIR(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.conv_first = nn.Conv2d(3, dim, 3, 1, 1)

        self.blocks = nn.Sequential(
            SwinBlock(dim, False),
            SwinBlock(dim, True),
            SwinBlock(dim, False),
            SwinBlock(dim, True),
        )

        self.conv_mid = nn.Conv2d(dim, dim, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * (SCALE ** 2), 3, 1, 1),
            nn.PixelShuffle(SCALE),
            nn.Conv2d(dim, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = self.conv_first(x)
        res = x
        x = self.blocks(x)
        x = self.conv_mid(x)
        x = x + res
        return self.upsample(x)

# ===============================
# LOAD MODEL
# ===============================
model = MiniSwinIR().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print("\nModel Loaded Successfully")
print(f"Total Parameters: {total_params/1e6:.2f} Million")

# ===============================
# SELECT HR IMAGE
# ===============================
Tk().withdraw()
hr_path = askopenfilename(title="Select HR Image")
if hr_path == "":
    print("No image selected.")
    exit()

hr_image = Image.open(hr_path).convert("RGB")
w, h = hr_image.size

# ✅ PRINT HR RESOLUTION
print("\n===== INPUT RESOLUTIONS =====")
print(f"HR Resolution : {w} x {h}")

# ===============================
# CREATE LR IMAGE
# ===============================
lr_image = hr_image.resize((w//SCALE, h//SCALE), Image.BICUBIC)
lw, lh = lr_image.size

# ✅ PRINT LR RESOLUTION
print(f"LR Resolution : {lw} x {lh}")

transform = transforms.ToTensor()
lr_tensor = transform(lr_image).unsqueeze(0).to(DEVICE)

# ===============================
# INFERENCE
# ===============================
start = time.time()
with torch.no_grad():
    sr_tensor = model(lr_tensor)
end = time.time()

sr_tensor = torch.clamp(sr_tensor, 0, 1)

# Convert tensor → numpy
sr_np = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
sr_np = (sr_np * 255).clip(0, 255).astype(np.uint8)

# Bilateral filter (remove ringing)
sr_np = cv2.bilateralFilter(sr_np, d=5, sigmaColor=50, sigmaSpace=50)

# Back to PIL
sr_image = Image.fromarray(sr_np)

# ===============================
# METRICS
# ===============================
hr_tensor = transform(hr_image).unsqueeze(0).to(DEVICE)

sr_tensor_filtered = torch.from_numpy(sr_np.transpose(2, 0, 1)).float() / 255.0
sr_tensor_filtered = sr_tensor_filtered.unsqueeze(0).to(DEVICE)

sr_resized = F.interpolate(sr_tensor_filtered, size=hr_tensor.shape[-2:], mode='bilinear')

psnr = calculate_psnr(sr_resized, hr_tensor)
ssim = calculate_ssim(sr_resized, hr_tensor)

# ✅ PRINT SR RESOLUTION
sw, sh = sr_image.size

print("\n===== OUTPUT =====")
print(f"SR Resolution : {sw} x {sh}")
print(f"Inference Time : {(end-start)*1000:.2f} ms")
print(f"PSNR : {psnr.item():.2f} dB")
print(f"SSIM : {ssim.item():.4f}")

# ===============================
# PATCHES
# ===============================
def extract_patch(img, patch_size=64):
    w, h = img.size
    cx, cy = w // 2, h // 2
    return img.crop((cx - patch_size//2, cy - patch_size//2,
                     cx + patch_size//2, cy + patch_size//2))

lr_patch = extract_patch(lr_image, 64)
sr_patch = extract_patch(sr_image, 64 * SCALE)
lr_patch_up = lr_patch.resize(sr_patch.size, Image.BICUBIC)
hr_patch = extract_patch(hr_image, 64 * SCALE)
# ===============================
# DISPLAY (ONLY RESOLUTION)
# ===============================

lr_res_text = f"{lw} x {lh}"
sr_res_text = f"{sw} x {sh}"
hr_res_text = f"{w} x {h}"

# ✅ Fixed top-left placement (same for all)
def put_res(ax, text):
    ax.text(
        0.01, 0.99, text,
        transform=ax.transAxes,
        fontsize=11,
        color='white',
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='black', alpha=0.7, pad=3)
    )

plt.figure(figsize=(15, 9))

# LR Image
ax1 = plt.subplot(2,3,1)
ax1.imshow(lr_image)
ax1.set_title("LR Input")
ax1.axis("off")
put_res(ax1, lr_res_text)

# SR Image
ax2 = plt.subplot(2,3,2)
ax2.imshow(sr_image)
ax2.set_title("SR Output")
ax2.axis("off")
put_res(ax2, sr_res_text)

# HR Image
ax3 = plt.subplot(2,3,3)
ax3.imshow(hr_image)
ax3.set_title("Ground Truth HR")
ax3.axis("off")
put_res(ax3, hr_res_text)

# Patches (no text)
plt.subplot(2,3,4)
plt.imshow(lr_patch_up)
plt.title("LR Patch (Bicubic)")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(sr_patch)
plt.title("SR Patch")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(hr_patch)
plt.title("HR Patch")
plt.axis("off")

plt.tight_layout()
plt.show()
