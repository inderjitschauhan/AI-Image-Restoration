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

# ================= CONFIG =================
TRAIN_DIR = r"data"
DNCNN_DIR = r"weights/dncnn"
SWIN_PATH = r"weights/sr/best_swinir_dn.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= UTIL =================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_model_size_kb(path):
    return os.path.getsize(path) / 1024

# ================= NAME CLEANING =================
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
def model_priority(name):
    name = name.lower()

    if "original" in name:
        return 0
    elif "noisy" in name:
        return 0.5
    elif "dncnn-8" in name:
        return 1
    elif "dncnn-17" in name:
        return 2
    elif "lr-sr" in name or "lrsr" in name:
        return 3
    elif "swinir" in name:
        return 4
    else:
        return 5

def sub_priority(name):
    name = name.lower()
    if "gaussian" in name:
        return 0
    elif "real" in name:
        return 1
    else:
        return 2

# ================= DNCNN =================
class DnCNN(nn.Module):
    def __init__(self, depth=17, channels=64, use_bn=False):
        super().__init__()
        layers = [nn.Conv2d(3, channels, 3, 1, 1), nn.ReLU(inplace=True)]

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(channels, channels, 3, 1, 1))
            if use_bn:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(channels, 3, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ================= SWINIR =================
class SwinBlock(nn.Module):
    def __init__(self, dim, shift=False):
        super().__init__()
        self.ws = 8
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 2, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def window_partition(self, x, ws):
        B, C, H, W = x.shape
        x = x.view(B, C, H // ws, ws, W // ws, ws)
        return x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, C)

    def window_reverse(self, windows, ws, H, W):
        B = int(windows.shape[0] / (H * W / ws / ws))
        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        return x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        if self.shift:
            x = torch.roll(x, (-ws // 2, -ws // 2), (2, 3))

        windows = self.window_partition(x, ws)
        windows = self.norm1(windows)

        attn, _ = self.attn(windows, windows, windows)
        windows = windows + attn
        windows = windows + self.mlp(self.norm2(windows))

        x = self.window_reverse(windows, ws, Hp, Wp)

        if self.shift:
            x = torch.roll(x, (ws // 2, ws // 2), (2, 3))

        return x[:, :, :H, :W]

class SwinIR_DN(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.conv_first = nn.Conv2d(3, dim, 3, 1, 1)

        self.blocks = nn.Sequential(
            SwinBlock(dim, False),
            SwinBlock(dim, True),
            SwinBlock(dim, False),
            SwinBlock(dim, True),
        )

        self.conv_mid = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(dim, 3, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        res = x
        x = self.blocks(x)
        x = self.conv_mid(x)
        x = x + res
        return self.conv_last(x)

# ================= LOAD =================
def load_dncnn(path):
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    state = state.get("model_state_dict", state)

    convs = [v for v in state.values() if len(v.shape) == 4]
    depth = len(convs)
    channels = convs[0].shape[0]
    use_bn = any("running_mean" in k for k in state.keys())

    model = DnCNN(depth, channels, use_bn).to(DEVICE)
    model.load_state_dict(state, strict=False)
    return model.eval()

def load_swinir(path):
    model = SwinIR_DN().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model.eval()

# ================= METRICS =================
def calculate_psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 100 if mse == 0 else 20 * log10(1.0 / np.sqrt(mse))

def calculate_ssim(a, b):
    return ssim(a, b, channel_axis=2, data_range=1.0)

# ================= IMAGE =================
def get_random_image(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))

    path = random.choice(paths)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def add_noise(img, sigma):
    noise = np.random.normal(0, sigma / 255.0, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1).astype(np.float32)

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

# ================= ADVANCED PLOTTING =================
def plot_results_full(df):
    for sigma in sorted(df["Sigma (σ)"].unique()):
        df_sigma = df[df["Sigma (σ)"] == sigma]

        models = df_sigma["Model"].values
        psnr = df_sigma["PSNR (dB)"].astype(float).values
        ssim_vals = df_sigma["SSIM"].astype(float).values

        time_vals = df_sigma["Time (s)"].astype(float).values
        params = df_sigma["Params (M)"].replace("NA", np.nan).astype(float).values
        size = df_sigma["Size (KB)"].replace("NA", np.nan).astype(float).values

        x = np.arange(len(models))
        width = 0.2  # narrow bars

        plt.figure(figsize=(16, 7))

        # -------- BARS --------
        bars1 = plt.bar(x - width/2, psnr, width, label="PSNR")
        bars2 = plt.bar(x + width/2, ssim_vals * 100, width, label="SSIM x100")

        # -------- ANNOTATIONS --------
        for i in range(len(models)):
            text = ""

            # Time
            text += f"T:{time_vals[i]:.3f}s\n"

            # Params
            if not np.isnan(params[i]):
                text += f"P:{params[i]:.2f}M\n"

            # Size
            if not np.isnan(size[i]):
                text += f"S:{size[i]:.0f}KB"

            plt.text(
                x[i],
                psnr[i] + 1,   # position above bar
                text,
                ha='center',
                fontsize=8
            )

        # -------- AXIS --------
        plt.xticks(x, models, rotation=40, ha='right')
        plt.title(f"Model Comparison (Sigma = {sigma})")
        plt.ylabel("PSNR / SSIM")
        plt.legend()

        plt.tight_layout()
        plt.show()
import matplotlib.pyplot as plt
import numpy as np

# ================= FINAL PLOT WITH DATA LABELS =================
def plot_results_stacked(df):
    sigmas = sorted(df["Sigma (σ)"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    for idx, sigma in enumerate(sigmas):
        ax = axes[idx]
        df_sigma = df[df["Sigma (σ)"] == sigma]

        models = df_sigma["Model"].values
        psnr = df_sigma["PSNR (dB)"].astype(float).values
        ssim_vals = df_sigma["SSIM"].astype(float).values

        time_vals = df_sigma["Time (s)"].astype(float).values
        params = df_sigma["Params (M)"].replace("NA", np.nan).astype(float).values
        size = df_sigma["Size (KB)"].replace("NA", np.nan).astype(float).values

        x = np.arange(len(models))
        width = 0.2

        # -------- BARS --------
        bars_psnr = ax.bar(x - width/2, psnr, width, label="PSNR")
        bars_ssim = ax.bar(x + width/2, ssim_vals * 100, width, label="SSIM x100")

        # -------- DATA LABELS (ON BARS) --------
        for bar in bars_psnr:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.2,
                f"{height:.2f}",
                ha='center',
                fontsize=9,
                fontweight='bold'
            )

        for bar, val in zip(bars_ssim, ssim_vals):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.2,
                f"{val:.3f}",   # show real SSIM (not x100)
                ha='center',
                fontsize=9,
                fontweight='bold'
            )

        # -------- T / P / S ANNOTATIONS --------
        for i in range(len(models)):
            text = f"T:{time_vals[i]:.3f}s\n"

            if not np.isnan(params[i]):
                text += f"P:{params[i]:.2f}M\n"

            if not np.isnan(size[i]):
                text += f"S:{size[i]:.0f}KB"

            ax.text(
                x[i],
                psnr[i] + 15,   # slightly above data labels
                text,
                ha='center',
                fontsize=10,
                fontweight='bold'
            )

        # -------- AXIS --------
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha='right')
        ax.set_title(f"Sigma = {sigma}", fontsize=14, fontweight='bold')
        ax.set_ylabel("PSNR / SSIM")
        ax.legend()

    plt.tight_layout()
    plt.show()
           
# ================= MAIN =================
def main():
    img = get_random_image(TRAIN_DIR)

    h, w = img.shape[:2]
    img = img[:(h // 8) * 8, :(w // 8) * 8]

    sigmas = [25, 50]
    results = []

    swin_model = load_swinir(SWIN_PATH)

    for sigma in sigmas:
        noisy = add_noise(img, sigma)
        noisy_tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

        # ----- ORIGINAL -----
        results.append({
            "Model": "Original (GT)",
            "PSNR": 100.0,
            "SSIM": 1.0,
            "Time": 0.0,
            "Sigma": sigma,
            "Params": None,
            "Size": None
        })

        # ----- NOISY -----
        results.append({
            "Model": f"Noisy-{sigma}",
            "PSNR": calculate_psnr(img, noisy),
            "SSIM": calculate_ssim(img, noisy),
            "Time": 0.0,
            "Sigma": sigma,
            "Params": None,
            "Size": None
        })

        # ----- DNCNN -----
        for f in os.listdir(DNCNN_DIR):
            if not f.endswith(".pth") or f"_{sigma}" not in f.lower():
                continue

            model = load_dncnn(os.path.join(DNCNN_DIR, f))

            with torch.no_grad():
                start = time.perf_counter()
                pred = model(noisy_tensor)
                out = noisy_tensor - pred
                end = time.perf_counter()

            out_np = out.squeeze().permute(1, 2, 0).cpu().numpy()
            out_np = np.clip(out_np, 0, 1)

            results.append({
                "Model": clean_name(f),
                "PSNR": calculate_psnr(img, out_np),
                "SSIM": calculate_ssim(img, out_np),
                "Time": end - start,
                "Sigma": sigma,
                "Params": count_parameters(model),
                "Size": get_model_size_kb(os.path.join(DNCNN_DIR, f))
            })

        # ----- SWINIR -----
        with torch.no_grad():
            start = time.perf_counter()
            pred = swin_model(noisy_tensor)
            out = noisy_tensor - pred
            end = time.perf_counter()

        out_np = out.squeeze().permute(1, 2, 0).cpu().numpy()
        out_np = np.clip(out_np, 0, 1)

        results.append({
            "Model": f"SwinIR-DN-{sigma}",
            "PSNR": calculate_psnr(img, out_np),
            "SSIM": calculate_ssim(img, out_np),
            "Time": end - start,
            "Sigma": sigma,
            "Params": count_parameters(swin_model),
            "Size": get_model_size_kb(SWIN_PATH)
        })

    # ================= FINAL TABLE =================
    df = pd.DataFrame(results).drop_duplicates(subset=["Model", "Sigma"])

    df["Priority"] = df["Model"].apply(model_priority)
    df["SubPriority"] = df["Model"].apply(sub_priority)

    df = df.sort_values(by=["Sigma", "Priority", "SubPriority"])

    df_display = df.rename(columns={
        "PSNR": "PSNR (dB)",
        "SSIM": "SSIM",
        "Time": "Time (s)",
        "Sigma": "Sigma (σ)",
        "Params": "Params (M)",
        "Size": "Size (KB)"
    }).drop(columns=["Priority", "SubPriority"])

    # ----- FORMATTING -----
    df_display["PSNR (dB)"] = df_display["PSNR (dB)"].map(lambda x: f"{x:.3f}")
    df_display["SSIM"] = df_display["SSIM"].map(lambda x: f"{x:.4f}")
    df_display["Time (s)"] = df_display["Time (s)"].map(lambda x: f"{x:.4f}")

    df_display["Params (M)"] = df_display["Params (M)"].map(
        lambda x: "NA" if pd.isna(x) else f"{x:.2f}"
    )

    df_display["Size (KB)"] = df_display["Size (KB)"].map(
        lambda x: "NA" if pd.isna(x) else f"{x:.1f}"
    )

    # ----- PRINT -----
    for sigma in sorted(df_display["Sigma (σ)"].unique()):
        df_sigma = df_display[df_display["Sigma (σ)"] == sigma]

        print(f"\n===== Sigma (σ) = {sigma} =====\n")
        print(df_sigma.to_string(index=False))
        print("\n")
    
    plot_results_stacked(df_display)

if __name__ == "__main__":
    main()
    