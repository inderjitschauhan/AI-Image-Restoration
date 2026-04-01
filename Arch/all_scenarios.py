import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ================= CONFIG =================
TRAIN_DIR = r"data"
DNCNN_DIR = r"weights/dncnn"
SWIN_PATH = r"weights/SR/best_mini_swinir.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)

# ================= FLAGS =================
USE_NOISE = False
USE_DNCNN = False
USE_SWIN = True

# ================= DNCNN =================
class DnCNN(nn.Module):
    def __init__(self, depth=17, channels=64, use_bn=False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(3, channels, 3, 1, 1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(channels, channels, 3, 1, 1))
            if use_bn:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(channels, 3, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ================= SWIN =================
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size,
               W // window_size, window_size)
    windows = x.permute(0,2,4,3,5,1).contiguous()
    return windows.view(-1, window_size*window_size, C)

def window_reverse(windows, window_size, H, W):
    denom = (H * W) / (window_size * window_size)
    if denom == 0:
        denom = 1
    B = int(windows.shape[0] / denom)

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

class MiniSwinIR(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.conv_first = nn.Conv2d(3, dim, 3, 1, 1)

        self.blocks = nn.Sequential(
        SwinBlock(dim, window_size=8, shift=False),
        SwinBlock(dim, window_size=8, shift=True),
        SwinBlock(dim, window_size=8, shift=False),
        SwinBlock(dim, window_size=8, shift=True),
        )

        self.conv_mid = nn.Conv2d(dim, dim, 3,1,1)

        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim*(4**2),3,1,1),
            nn.PixelShuffle(4),
            nn.Conv2d(dim,3,3,1,1)
        )

    def forward(self,x):
        x = self.conv_first(x)
        res = x
        x = self.blocks(x)
        x = self.conv_mid(x)
        x = x + res
        return self.upsample(x)

def load_swin_model(path):
    model = MiniSwinIR().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    return model.eval()

# ================= METRICS =================
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 100 if mse == 0 else 20 * log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2, data_range=1.0)

# ================= IMAGE =================
def get_random_image(folder):
    valid_images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg')) and "lr" in f.lower():
                valid_images.append(os.path.join(root, f))

    if not valid_images:
        raise ValueError("No LR images found!")

    path = random.choice(valid_images)

    print("Selected:", path)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.float32) / 255.0

# ================= NOISE =================
def add_real_noise(img):
    shot = np.random.uniform(0.005, 0.02)
    read = np.random.uniform(0.001, 0.01)

    noisy = np.random.poisson(img / shot) * shot
    noisy += np.random.normal(0, read, img.shape)

    return np.clip(noisy, 0., 1.).astype(np.float32)

# ================= LOAD DNCNN =================
def load_dncnn(path):
    state = torch.load(path, map_location=DEVICE)
    state = state.get("model_state_dict", state)

    convs = [v for v in state.values() if len(v.shape)==4]
    depth = len(convs)
    channels = convs[0].shape[0]
    use_bn = any("running_mean" in k for k in state.keys())

    model = DnCNN(depth, channels, use_bn).to(DEVICE)
    model.load_state_dict(state, strict=False)

    return model.eval()

# ================= DISPLAY =================
def show_all(clean, noisy, dncnn, final,
             psnr_noisy, ssim_noisy,
             psnr_dncnn, ssim_dncnn,
             psnr_final, ssim_final):

    plt.figure(figsize=(14,5))
    imgs = [clean, noisy, dncnn, final]
    titles = ["Original","Noisy","DnCNN","Final"]

    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(imgs[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.show()

# ================= MAIN =================
def main():

    clean = get_random_image(TRAIN_DIR)

    # -------- NOISE --------
    if USE_NOISE:
        noisy = add_real_noise(clean)
    else:
        noisy = clean.copy()

    noisy_tensor = torch.from_numpy(noisy).float().permute(2,0,1).unsqueeze(0).to(DEVICE)

    best_psnr = -1
    best_dncnn = noisy.copy()

    # -------- DNCNN --------
    if USE_DNCNN:
        for f in os.listdir(DNCNN_DIR):
            if f.endswith(".pth"):
                model = load_dncnn(os.path.join(DNCNN_DIR, f))

                with torch.no_grad():
                    pred = model(noisy_tensor)
                    out = noisy_tensor - pred

                img = out.squeeze().permute(1,2,0).cpu().numpy()
                img = np.clip(img,0,1)

                psnr = calculate_psnr(clean, img)

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_dncnn = img
    else:
        best_dncnn = noisy.copy()

    # -------- SWIN --------
    if USE_SWIN:
        lr = best_dncnn

        swin = load_swin_model(SWIN_PATH)
        inp = torch.from_numpy(lr).float().permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            final = swin(inp)

        final = final.squeeze().permute(1,2,0).cpu().numpy()
        final = np.clip(final,0,1)
    else:
        final = best_dncnn.copy()

    clean_resized = cv2.resize(clean, (final.shape[1], final.shape[0]))

    # -------- METRICS --------
    psnr_noisy = calculate_psnr(clean, noisy)
    ssim_noisy = calculate_ssim(clean, noisy)

    psnr_dncnn = calculate_psnr(clean, best_dncnn)
    ssim_dncnn = calculate_ssim(clean, best_dncnn)

    psnr_final = calculate_psnr(clean_resized, final)
    ssim_final = calculate_ssim(clean_resized, final)

    print("\n=========== RESULTS ===========")
    print(f"Noisy  → {psnr_noisy:.2f} | {ssim_noisy:.4f}")
    print(f"DnCNN  → {psnr_dncnn:.2f} | {ssim_dncnn:.4f}")
    print(f"Final  → {psnr_final:.2f} | {ssim_final:.4f}")

    show_all(clean, noisy, best_dncnn, final,
             psnr_noisy, ssim_noisy,
             psnr_dncnn, ssim_dncnn,
             psnr_final, ssim_final)

if __name__ == "__main__":
    main()