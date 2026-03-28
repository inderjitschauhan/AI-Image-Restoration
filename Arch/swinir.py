
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import math

# ===============================
# CONFIG
# ===============================
SCALE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "weights/SR/best_mini_swinir.pth"

# ===============================
# METRICS
# ===============================
def calculate_psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    return 10 * torch.log10(1.0 / (mse + 1e-8))

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
# MODEL
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
# LOAD MODEL
# ===============================
model = MiniSwinIR().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")

# ===============================
# PICK IMAGE
# ===============================
Tk().withdraw()
image_path = askopenfilename(title="Select LR Image")

if image_path == "":
    print("No image selected.")
    exit()

lr_image = Image.open(image_path).convert("RGB")
lr_w, lr_h = lr_image.size

transform = transforms.ToTensor()
lr_tensor = transform(lr_image).unsqueeze(0).to(DEVICE)

# ===============================
# INFERENCE
# ===============================
with torch.no_grad():
    sr_tensor = model(lr_tensor)

sr_tensor = sr_tensor.clamp(0,1)
sr_image = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
sr_w, sr_h = sr_image.size

# ===============================
# PRINT RESOLUTION
# ===============================
print("\n===== IMAGE STATS =====")
print(f"LR Resolution : {lr_w} x {lr_h}")
print(f"SR Resolution : {sr_w} x {sr_h}")

# ===============================
# CHECK FOR HR GROUND TRUTH
# ===============================
hr_path = image_path.replace("_LR4", "_HR")

if os.path.exists(hr_path):
    print("HR Ground Truth found. Calculating metrics...")

    hr_image = Image.open(hr_path).convert("RGB")
    hr_tensor = transform(hr_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        psnr = calculate_psnr(sr_tensor, hr_tensor)
        ssim = calculate_ssim(sr_tensor, hr_tensor)

    print(f"PSNR : {psnr.item():.4f} dB")
    print(f"SSIM : {ssim.item():.4f}")
else:
    print("No HR ground truth found. Skipping PSNR/SSIM.")

# ===============================
# SAVE OUTPUT
# ===============================
output_path = "restored_HR.png"
sr_image.save(output_path)
print(f"Restored image saved as {output_path}")

# ===============================
# DISPLAY
# ===============================
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Input LR Image")
plt.imshow(lr_image)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Restored HR Image")
plt.imshow(sr_image)
plt.axis("off")

plt.tight_layout()
plt.show()