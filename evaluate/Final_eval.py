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

# ================= PLOT STYLE =================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})

# ================= UTIL =================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_model_size_kb(path):
    return os.path.getsize(path) / 1024

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

# ================= BAR LABELS =================
def add_bar_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

# ================= FINAL PLOT =================
def plot_publication_figure(df):
    sigmas = sorted(df["Sigma"].unique())

    for sigma in sigmas:
        df_sigma = df[df["Sigma"] == sigma].copy()
        df_sigma["Priority"] = df_sigma.apply(model_priority, axis=1)
        df_sigma = df_sigma.sort_values(by=["Priority"])

        models = df_sigma["Model"]
        x = np.arange(len(models))

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f"Model Comparison (Noise σ = {sigma})", fontsize=16)

        def draw(ax, values, title, ylabel):
            bars = ax.bar(x, values)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=30, ha="right")
            ax.grid(alpha=0.3)
            add_bar_labels(ax, bars)

        # ================= METRICS =================
        draw(axes[0, 0], df_sigma["PSNR"], "PSNR", "dB")
        draw(axes[0, 1], df_sigma["SSIM"], "SSIM", "0–1")
        draw(axes[0, 2], df_sigma["Time"], "Inference Time", "seconds")
        draw(axes[1, 0], df_sigma["Params (M)"], "Parameters", "Millions")
        draw(axes[1, 1], df_sigma["Size (KB)"], "Model Size", "KB")

        score = (
            df_sigma["PSNR"] / df_sigma["PSNR"].max() +
            df_sigma["SSIM"] / df_sigma["SSIM"].max()
        ) / 2

        draw(axes[1, 2], score, "Normalized Score", "score")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# ================= MODELS =================
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

# ================= LOAD IMAGE =================
def get_random_image(folder):
    imgs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "lr" in f.lower():
                imgs.append(os.path.join(root, f))

    path = random.choice(imgs)
    print("Selected:", path)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma / 255.0, img.shape)
    return np.clip(img + noise, 0, 1).astype(np.float32)

# ================= METRICS =================
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 100 if mse == 0 else 20 * log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2, data_range=1.0)

# ================= LOAD DNCNN =================
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

# ================= MAIN =================
def main():
    clean = get_random_image(TRAIN_DIR)

    h, w = clean.shape[:2]
    clean = clean[:(h // 8) * 8, :(w // 8) * 8]

    results = []
    sigmas = [25, 50]

    for sigma in sigmas:
        noisy = add_gaussian_noise(clean, sigma)
        noisy_tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        best_psnr = -1
        best_dncnn = None

        for f in os.listdir(DNCNN_DIR):
            if not f.endswith(".pth") or f"_{sigma}" not in f.lower():
                continue

            model = load_dncnn(os.path.join(DNCNN_DIR, f))

            with torch.no_grad():
                start = time.time()
                pred = model(noisy_tensor)
                out = noisy_tensor - pred
                end = time.time()

            img = out.squeeze().permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)

            psnr = calculate_psnr(clean, img)
            ssim_val = calculate_ssim(clean, img)

            results.append({
                "Model": clean_name(f),
                "PSNR": psnr,
                "SSIM": ssim_val,
                "Time": end - start,
                "Sigma": sigma,
                "Params (M)": count_parameters(model),
                "Size (KB)": get_model_size_kb(os.path.join(DNCNN_DIR, f))
            })

            if psnr > best_psnr:
                best_psnr = psnr
                best_dncnn = img

    df = pd.DataFrame(results)
    df["Priority"] = df.apply(model_priority, axis=1)
    df = df.sort_values(by=["Sigma", "Priority"])

    print(df.to_string(index=False))

    plot_publication_figure(df)

if __name__ == "__main__":
    main()