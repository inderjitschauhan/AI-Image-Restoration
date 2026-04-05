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
# SWIN BLOCK (FIXED)
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

        # ✅ FIX: prevent zero division
        ws = min(self.ws, H, W)
        ws = max(ws, 1)

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
# MODEL
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

print("\nModel Loaded Successfully")

# ===============================
# LOAD IMAGE
# ===============================
Tk().withdraw()
hr_path = askopenfilename(title="Select HR Image")
hr_image = Image.open(hr_path).convert("RGB")

w, h = hr_image.size
lr_image = hr_image.resize((w // SCALE, h // SCALE), Image.BICUBIC)

print("\n===== INPUT RESOLUTIONS =====")
print(f"HR Resolution : {w} x {h}")
print(f"LR Resolution : {lr_image.size[0]} x {lr_image.size[1]}")

transform = transforms.ToTensor()
lr_tensor = transform(lr_image).unsqueeze(0).to(DEVICE)

# ===============================
# 🔥 2-STAGE SR (SAFE)
# ===============================
start = time.time()

with torch.no_grad():
    # First SR
    sr_tensor = model(lr_tensor)

    # Downscale safely
    sr_tensor = F.interpolate(
        sr_tensor,
        scale_factor=0.5,
        mode='bicubic',
        align_corners=False
    )

    # Second SR
    sr_tensor = model(sr_tensor)

end = time.time()

sr_tensor = torch.clamp(sr_tensor, 0, 1)

# ===============================
# CONVERT + FILTER
# ===============================
sr_np = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
sr_np = (sr_np * 255).clip(0, 255).astype(np.uint8)

sr_np = cv2.bilateralFilter(sr_np, 5, 50, 50)

sr_image = Image.fromarray(sr_np)

sw, sh = sr_image.size

print("\n===== OUTPUT =====")
print(f"SR Resolution : {sw} x {sh}")
print(f"Inference Time : {(end-start)*1000:.2f} ms")

# ===============================
# PATCHES
# ===============================
def extract_patch(img, patch_size=64):
    w, h = img.size
    cx, cy = w // 2, h // 2
    return img.crop((cx - patch_size//2, cy - patch_size//2,
                     cx + patch_size//2, cy + patch_size//2))

scale_factor = sw // lr_image.size[0]

lr_patch = extract_patch(lr_image, 64)
sr_patch = extract_patch(sr_image, 64 * scale_factor)
lr_patch_up = lr_patch.resize(sr_patch.size, Image.BICUBIC)
hr_patch = extract_patch(hr_image, 64 * scale_factor)

# ===============================
# DISPLAY
# ===============================
def put_res(ax, text):
    ax.text(0.01, 0.99, text,
            transform=ax.transAxes,
            fontsize=11,
            color='white',
            verticalalignment='top',
            bbox=dict(facecolor='black', alpha=0.7, pad=3))

plt.figure(figsize=(15, 9))

ax1 = plt.subplot(2,3,1)
ax1.imshow(lr_image)
ax1.set_title("LR")
ax1.axis("off")
put_res(ax1, f"{lr_image.size[0]} x {lr_image.size[1]}")

ax2 = plt.subplot(2,3,2)
ax2.imshow(sr_image)
ax2.set_title("SR (2-stage)")
ax2.axis("off")
put_res(ax2, f"{sw} x {sh}")

ax3 = plt.subplot(2,3,3)
ax3.imshow(hr_image)
ax3.set_title("HR")
ax3.axis("off")
put_res(ax3, f"{w} x {h}")

plt.subplot(2,3,4)
plt.imshow(lr_patch_up)
plt.title("LR Patch")
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
