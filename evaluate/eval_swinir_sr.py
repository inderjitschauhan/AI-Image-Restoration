import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import lpips

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "weights/sr/best_swinir_sr.pth"
SCALE = 4

# ===============================
# LPIPS
# ===============================
lpips_model = lpips.LPIPS(net='alex').to(DEVICE)

# ===============================
# SAFE WINDOW FUNCTIONS
# ===============================
def pad_to_window(x, ws):
    B, C, H, W = x.shape
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, H, W

# ===============================
# MODEL
# ===============================
class SwinBlock(nn.Module):
    def __init__(self, dim, window_size=8, shift=False):
        super().__init__()

        assert window_size > 0
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
        ws = max(1, self.window_size)

        x, H_orig, W_orig = pad_to_window(x, ws)
        _, _, Hp, Wp = x.shape

        if self.shift:
            x = torch.roll(x, shifts=(-ws//2, -ws//2), dims=(2,3))

        # Partition
        x_windows = x.view(B, C, Hp//ws, ws, Wp//ws, ws)
        x_windows = x_windows.permute(0,2,4,3,5,1).contiguous().view(-1, ws*ws, C)

        x_windows = self.norm1(x_windows)
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)

        x_windows = x_windows + attn_out
        x_windows = x_windows + self.mlp(self.norm2(x_windows))

        # Reverse
        x = x_windows.view(B, Hp//ws, Wp//ws, ws, ws, C)
        x = x.permute(0,5,1,3,2,4).contiguous().view(B, C, Hp, Wp)

        if self.shift:
            x = torch.roll(x, shifts=(ws//2, ws//2), dims=(2,3))

        return x[:, :, :H_orig, :W_orig]

class MiniSwinIR(nn.Module):
    def __init__(self, dim=96):
        super().__init__()

        self.conv_first = nn.Conv2d(3, dim, 3, 1, 1)

        self.blocks = nn.Sequential(
            SwinBlock(dim, 8, False),
            SwinBlock(dim, 8, True),
            SwinBlock(dim, 8, False),
            SwinBlock(dim, 8, True),
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

    return (((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) /
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))).mean()

# ===============================
# SELECT IMAGE
# ===============================
Tk().withdraw()
lr_path = askopenfilename(title="Select LR Image (_LR4)")

if "_LR4" not in lr_path:
    raise ValueError("Select a RealSR LR image")

hr_path = lr_path.replace("_LR4", "_HR")

if not os.path.exists(hr_path):
    raise FileNotFoundError("HR image not found")

# ===============================
# LOAD
# ===============================
transform = transforms.ToTensor()

lr_img = Image.open(lr_path).convert("RGB")
hr_img = Image.open(hr_path).convert("RGB")

lr_tensor = transform(lr_img).unsqueeze(0).to(DEVICE)
hr_tensor = transform(hr_img).unsqueeze(0).to(DEVICE)

# ===============================
# MODEL LOAD
# ===============================
model = MiniSwinIR().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===============================
# INFERENCE
# ===============================
with torch.no_grad():
    sr = model(lr_tensor)
    sr = torch.clamp(sr, 0, 1)

# ===============================
# METRICS
# ===============================
psnr_val = calculate_psnr(sr, hr_tensor).item()
ssim_val = calculate_ssim(sr, hr_tensor).item()

sr_lp = (sr * 2 - 1)
hr_lp = (hr_tensor * 2 - 1)
lpips_val = lpips_model(sr_lp, hr_lp).item()

# ===============================
# TEXT INFO
# ===============================
lr_h, lr_w = lr_tensor.shape[-2:]
sr_h, sr_w = sr.shape[-2:]

text_info = f"PSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}\nLPIPS: {lpips_val:.4f}\nRes: {lr_w}x{lr_h} -> {sr_w}x{sr_h}"

# ===============================
# CONVERT IMAGE
# ===============================
sr_img = (sr.squeeze().cpu().numpy().transpose(1,2,0) * 255).astype("uint8")

# ===============================
# DISPLAY
# ===============================
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.imshow(lr_img)
plt.title("Input LR")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sr_img)
plt.axis("off")

plt.text(10, 30, text_info, color='white',
         fontsize=12,
         bbox=dict(facecolor='black', alpha=0.7))

plt.title("Super-Resolved Output")

plt.tight_layout()
plt.savefig("stable_sr_output.png", dpi=300, bbox_inches='tight')
plt.show()