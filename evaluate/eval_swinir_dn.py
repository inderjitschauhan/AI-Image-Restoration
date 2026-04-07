import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import os

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGMA = 50  #  CHANGE to 50 when needed
MODEL_PATH = "weights/sr/best_swinir_dn.pth"

# ===============================
# REAL NOISE (MATCH TRAINING)
# ===============================
def add_real_noise(img, sigma):
    img = img / 255.0

    shot = np.random.poisson(img * 255.0) / 255.0 - img
    read = np.random.normal(0, sigma / 255.0, img.shape)

    noisy = img + shot + 0.5 * read
    noisy = np.clip(noisy, 0, 1)

    return (noisy * 255).astype(np.uint8)

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

    def window_partition(self, x, ws):
        B,C,H,W = x.shape
        x = x.view(B, C, H//ws, ws, W//ws, ws)
        return x.permute(0,2,4,3,5,1).contiguous().view(-1, ws*ws, C)

    def window_reverse(self, windows, ws, H, W):
        B = int(windows.shape[0] / (H*W/ws/ws))
        x = windows.view(B, H//ws, W//ws, ws, ws, -1)
        return x.permute(0,5,1,3,2,4).contiguous().view(B, -1, H, W)

    def forward(self, x):
        B,C,H,W = x.shape
        ws = self.ws

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0,pad_w,0,pad_h))
        _,_,Hp,Wp = x.shape

        if self.shift:
            x = torch.roll(x, (-ws//2,-ws//2),(2,3))

        windows = self.window_partition(x, ws)
        windows = self.norm1(windows)

        attn,_ = self.attn(windows, windows, windows)
        windows = windows + attn
        windows = windows + self.mlp(self.norm2(windows))

        x = self.window_reverse(windows, ws, Hp, Wp)

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
        return self.conv_last(x)  # predicts noise

# ===============================
# LOAD IMAGE
# ===============================
Tk().withdraw()
path = askopenfilename(title="Select Image")

img = Image.open(path).convert("RGB")
img_np = np.array(img).astype(np.float32)

transform = transforms.ToTensor()
clean_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ===============================
# ADD NOISE
# ===============================
noisy_np = add_real_noise(img_np.copy(), SIGMA)
noisy_img = Image.fromarray(noisy_np)
noisy_tensor = transform(noisy_img).unsqueeze(0).to(DEVICE)

# ===============================
# LOAD MODEL
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Loading model:", MODEL_PATH)

model = SwinIR_DN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===============================
# INFERENCE
# ===============================
with torch.no_grad():
    pred_noise = model(noisy_tensor)
    denoised = torch.clamp(noisy_tensor - pred_noise, 0, 1)

# ===============================
# METRICS
# ===============================
psnr_noisy = calculate_psnr(noisy_tensor, clean_tensor).item()
psnr_denoised = calculate_psnr(denoised, clean_tensor).item()

ssim_noisy = calculate_ssim(noisy_tensor, clean_tensor).item()
ssim_denoised = calculate_ssim(denoised, clean_tensor).item()

print(f"Noisy   → PSNR: {psnr_noisy:.2f}, SSIM: {ssim_noisy:.4f}")
print(f"Denoised→ PSNR: {psnr_denoised:.2f}, SSIM: {ssim_denoised:.4f}")

# ===============================
# CONVERT OUTPUT
# ===============================
den_np = denoised.squeeze().cpu().numpy().transpose(1,2,0)
den_np = (den_np * 255).astype(np.uint8)
den_img = Image.fromarray(den_np)

# ===============================
# PLOT (WITH METRICS ON IMAGE)
# ===============================
plt.figure(figsize=(12,6))

# ---- NOISY ----
plt.subplot(1,2,1)
plt.imshow(noisy_img)
plt.axis("off")
plt.text(
    10, 25,
    f"Noisy (σ={SIGMA})\nPSNR: {psnr_noisy:.2f} dB\nSSIM: {ssim_noisy:.4f}",
    color='white',
    fontsize=12,
    bbox=dict(facecolor='black', alpha=0.6)
)

# ---- DENOISED ----
plt.subplot(1,2,2)
plt.imshow(den_img)
plt.axis("off")
plt.text(
    10, 25,
    f"Denoised\nPSNR: {psnr_denoised:.2f} dB\nSSIM: {ssim_denoised:.4f}",
    color='white',
    fontsize=12,
    bbox=dict(facecolor='black', alpha=0.6)
)

plt.tight_layout()

# Save
plt.savefig(f"swinir_result_sigma_{SIGMA}.png", dpi=300, bbox_inches='tight')

plt.show()