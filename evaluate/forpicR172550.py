import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGMA_LIST = [25, 50]

MODEL_PATHS = {
    25: "weights/dncnn/Dncnn_17_Real_Best_25.pth",
    50: "weights/dncnn/Dncnn_17_Real_Best_50.pth"
}

# ===============================
# MODEL (MATCH TRAINING EXACTLY)
# ===============================
class DnCNN(nn.Module):
    def __init__(self, depth=17, channels=64):
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
# REAL NOISE (MATCH TRAINING)
# ===============================
def add_real_noise(img, sigma):
    img = img / 255.0

    # Shot noise (Poisson)
    shot = np.random.poisson(img * 255.0) / 255.0 - img

    # Read noise (Gaussian)
    read = np.random.normal(0, sigma / 255.0, img.shape)

    noisy = img + shot + read
    noisy = np.clip(noisy, 0, 1)

    return (noisy * 255).astype(np.uint8)

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
# PLOT SETUP
# ===============================
plt.figure(figsize=(12, 10))

# ===============================
# MAIN LOOP
# ===============================
for i, sigma in enumerate(SIGMA_LIST):

    print(f"\n===== SIGMA {sigma} =====")

    # Add real noise
    noisy_np = add_real_noise(img_np.copy(), sigma)
    noisy_img = Image.fromarray(noisy_np)

    noisy_tensor = transform(noisy_img).unsqueeze(0).to(DEVICE)

    # Load model
    model = DnCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATHS[sigma], map_location=DEVICE))
    model.eval()

    # Inference
    with torch.no_grad():
        pred_noise = model(noisy_tensor)
        denoised = torch.clamp(noisy_tensor - pred_noise, 0, 1)

    # Metrics
    psnr_noisy = calculate_psnr(noisy_tensor, clean_tensor)
    psnr_denoised = calculate_psnr(denoised, clean_tensor)

    ssim_noisy = calculate_ssim(noisy_tensor, clean_tensor)
    ssim_denoised = calculate_ssim(denoised, clean_tensor)

    print(f"Noisy   → PSNR: {psnr_noisy:.2f} | SSIM: {ssim_noisy:.4f}")
    print(f"Denoised→ PSNR: {psnr_denoised:.2f} | SSIM: {ssim_denoised:.4f}")

    # Convert output
    den_np = denoised.squeeze().cpu().numpy().transpose(1,2,0)
    den_np = (den_np * 255).astype(np.uint8)
    den_img = Image.fromarray(den_np)

    # ===============================
    # 2x2 GRID
    # ===============================

    # Noisy
    ax1 = plt.subplot(2, 2, i*2 + 1)
    ax1.imshow(noisy_img)
    ax1.set_title(f"Noisy σ={sigma}\nPSNR: {psnr_noisy:.2f}, SSIM: {ssim_noisy:.4f}")
    ax1.axis("off")

    # Denoised
    ax2 = plt.subplot(2, 2, i*2 + 2)
    ax2.imshow(den_img)
    ax2.set_title(f"DnCNN-17 σ={sigma}\nPSNR: {psnr_denoised:.2f}, SSIM: {ssim_denoised:.4f}")
    ax2.axis("off")

# ===============================
# SAVE + SHOW
# ===============================
plt.tight_layout()
#plt.savefig("results/dncnn17_real_25_50.png", dpi=300, bbox_inches='tight')
plt.show()
