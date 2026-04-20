import time
import pandas as pd
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
SWIN_SR_PATH = r"weights/SR/best_swinir_sr.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

# ================= CLEAN NAME =================
def clean_name(name):
    name_low = name.lower()

    if "lrsr" in name_low:
        base = "DnCNN-Real-LR-SR"
    elif "gaussian" in name_low:
        base = "DnCNN-Gaussian"
    elif "real" in name_low:
        base = "DnCNN-Real"
    else:
        base = "DnCNN"

    if "_17_" in name_low:
        base = base.replace("DnCNN", "DnCNN-17")
    elif "_8_" in name_low:
        base = base.replace("DnCNN", "DnCNN-8")

    if "_25" in name_low:
        base += "-25"
    elif "_50" in name_low:
        base += "-50"

    return base

# ================= SORTING =================
def model_priority(row):
    name = row["Model"].lower()

    if "noisy" in name:
        return 0
    elif "dncnn-8" in name:
        return 1
    elif "dncnn-17" in name:
        return 2
    elif "dncnn" in name:
        return 3
    elif "swinir" in name:
        return 99
    else:
        return 50

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

# ================= SWIN HELPERS =================
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size,
               W // window_size, window_size)
    windows = x.permute(0,2,4,3,5,1).contiguous()
    return windows.view(-1, window_size*window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0,5,1,3,2,4).contiguous()
    return x.view(B, -1, H, W)

# ================= SWIN BLOCK =================
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

        # ---- PAD ----
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[2], x.shape[3]

        # ---- SHIFT ----
        if self.shift:
            x = torch.roll(x, shifts=(-ws//2, -ws//2), dims=(2,3))

        # ---- WINDOW ----
        windows = window_partition(x, ws)
        windows = self.norm1(windows)

        attn_out,_ = self.attn(windows, windows, windows)
        windows = windows + attn_out
        windows = windows + self.mlp(self.norm2(windows))

        # ---- REVERSE ----
        x = window_reverse(windows, ws, Hp, Wp)

        if self.shift:
            x = torch.roll(x, shifts=(ws//2, ws//2), dims=(2,3))

        # ---- CROP BACK ----
        return x[:, :, :H, :W]

# ================= SWIN MODEL =================
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
    state = torch.load(path, map_location=DEVICE, weights_only=True)
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
    imgs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg')) and "lr" in f.lower():
                imgs.append(os.path.join(root, f))

    path = random.choice(imgs)
    print("Selected:", path)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.float32) / 255.0

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma/255.0, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0., 1.).astype(np.float32)

# ================= LOAD DNCNN =================
def load_dncnn(path):
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    state = state.get("model_state_dict", state)

    convs = [v for v in state.values() if len(v.shape)==4]
    depth = len(convs)
    channels = convs[0].shape[0]
    use_bn = any("running_mean" in k for k in state.keys())

    model = DnCNN(depth, channels, use_bn).to(DEVICE)
    model.load_state_dict(state, strict=False)

    return model.eval()

# ================= PLOT =================
def plot_results(df):
    sigmas = sorted(df["Sigma"].unique())

    for sigma in sigmas:
        df_sigma = df[df["Sigma"] == sigma].copy()
        df_sigma = df_sigma.sort_values(by="Priority")

        models = df_sigma["Model"]

        # PSNR
        plt.figure()
        plt.bar(models, df_sigma["PSNR"])
        plt.xticks(rotation=45, ha='right')
        plt.title(f"PSNR (σ={sigma})")
        plt.tight_layout()
        plt.show()

        # SSIM
        plt.figure()
        plt.bar(models, df_sigma["SSIM"])
        plt.xticks(rotation=45, ha='right')
        plt.title(f"SSIM (σ={sigma})")
        plt.tight_layout()
        plt.show()

        # TIME
        plt.figure()
        plt.bar(models, df_sigma["Time"])
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Inference Time (σ={sigma})")
        plt.tight_layout()
        plt.show()

# ================= MAIN =================
def main():
    clean = get_random_image(TRAIN_DIR)

    # optional: make divisible by 8 (extra safety)
    h, w = clean.shape[:2]
    clean = clean[:(h//8)*8, :(w//8)*8]

    results = []
    sigmas = [25, 50]

    swin = load_swin_model(SWIN_SR_PATH)

    for sigma in sigmas:
        print(f"\n=========== SIGMA {sigma} ===========")

        noisy = add_gaussian_noise(clean, sigma)
        noisy_tensor = torch.from_numpy(noisy).permute(2,0,1).unsqueeze(0).to(DEVICE)

        best_psnr = -1
        best_dncnn = None

        for f in os.listdir(DNCNN_DIR):
            if not f.endswith(".pth"):
                continue
            if f"_{sigma}" not in f.lower():
                continue

            model = load_dncnn(os.path.join(DNCNN_DIR, f))

            with torch.no_grad():
                start = time.time()
                pred = model(noisy_tensor)
                out = noisy_tensor - pred
                end = time.time()

            img = out.squeeze().permute(1,2,0).cpu().numpy()
            img = np.clip(img,0,1)

            psnr = calculate_psnr(clean, img)
            ssim_val = calculate_ssim(clean, img)

            results.append({
                "Model": clean_name(f),
                "PSNR": psnr,
                "SSIM": ssim_val,
                "Time": end-start,
                "Stage": f"DnCNN σ{sigma}",
                "Sigma": sigma
            })

            if psnr > best_psnr:
                best_psnr = psnr
                best_dncnn = img

        # ---- SwinIR ----
        inp = torch.from_numpy(best_dncnn).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            start = time.time()
            final = swin(inp)
            end = time.time()

        final = final.squeeze().permute(1,2,0).cpu().numpy()
        final = np.clip(final,0,1)

        clean_resized = cv2.resize(clean, (final.shape[1], final.shape[0]))

        results.append({
            "Model": f"SwinIR σ{sigma}",
            "PSNR": calculate_psnr(clean_resized, final),
            "SSIM": calculate_ssim(clean_resized, final),
            "Time": end-start,
            "Stage": "Final",
            "Sigma": sigma
        })

        # ---- Baseline ----
        results.append({
            "Model": f"Noisy σ{sigma}",
            "PSNR": calculate_psnr(clean, noisy),
            "SSIM": calculate_ssim(clean, noisy),
            "Time": 0,
            "Stage": "Baseline",
            "Sigma": sigma
        })

    # ===== FINAL SORT =====
    df = pd.DataFrame(results)
    df["Priority"] = df.apply(model_priority, axis=1)
    df = df.sort_values(by=["Sigma", "Priority"])

    print("\n=========== FINAL RESULTS ===========\n")
    print(df.drop(columns=["Priority"]).to_string(index=False))

    plot_results(df)

# ================= RUN =================
if __name__ == "__main__":
    main()