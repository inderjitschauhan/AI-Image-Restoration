import time
import pandas as pd
import os
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
SWIN_PATH = r"weights/sr/best_swinir_dn.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= UTIL =================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_model_size_kb(path):
    return os.path.getsize(path) / 1024

# ================= METRICS =================
def calculate_psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 100 if mse == 0 else 20 * log10(1.0 / np.sqrt(mse))

def calculate_ssim(a, b):
    return ssim(a, b, channel_axis=2, data_range=1.0)

# ================= IMAGE LOADER =================
def get_all_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))
    return paths

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def add_noise(img, sigma):
    noise = np.random.normal(0, sigma / 255.0, img.shape)
    return np.clip(img + noise, 0, 1).astype(np.float32)
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

def load_swinir(path):
    model = SwinIR_DN().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model.eval()

# ================= MAIN =================
def main():

    image_paths = get_all_images(TRAIN_DIR)
    sigmas = [25, 50]

    swin_model = load_swinir(SWIN_PATH)

    final_results = []

    for img_idx, img_path in enumerate(image_paths):

        img = load_image(img_path)

        h, w = img.shape[:2]
        img = img[:(h // 8) * 8, :(w // 8) * 8]

        print(f"\n================ IMAGE {img_idx+1}/{len(image_paths)} =================")
        print(f"File: {img_path}")

        for sigma in sigmas:

            noisy = add_noise(img, sigma)
            noisy_tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

            results = []

            # -------- ORIGINAL --------
            results.append(("Original", 100.0, 1.0))

            # -------- NOISY --------
            results.append((
                f"Noisy-{sigma}",
                calculate_psnr(img, noisy),
                calculate_ssim(img, noisy)
            ))

            # -------- DNCNN --------
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

                psnr = calculate_psnr(img, out_np)
                ssim_val = calculate_ssim(img, out_np)

                results.append((f, psnr, ssim_val))

            # -------- SWINIR --------
            with torch.no_grad():
                start = time.perf_counter()
                pred = swin_model(noisy_tensor)
                out = noisy_tensor - pred
                end = time.perf_counter()

            out_np = out.squeeze().permute(1, 2, 0).cpu().numpy()
            out_np = np.clip(out_np, 0, 1)

            psnr = calculate_psnr(img, out_np)
            ssim_val = calculate_ssim(img, out_np)

            results.append((f"SwinIR-{sigma}", psnr, ssim_val))

            # ================= BEST MODEL =================
            best = max(results, key=lambda x: x[1])

            print(f"\n--- Sigma {sigma} Results ---")
            for r in results:
                print(f"{r[0]:20s} | PSNR: {r[1]:.3f} | SSIM: {r[2]:.4f}")

            print(f"\n🏆 BEST MODEL: {best[0]} | PSNR: {best[1]:.3f} | SSIM: {best[2]:.4f}")

            final_results.append({
                "Image": os.path.basename(img_path),
                "Sigma": sigma,
                "Best Model": best[0],
                "Best PSNR": best[1],
                "Best SSIM": best[2]
            })

    # ================= FINAL SUMMARY =================
    df = pd.DataFrame(final_results)

    print("\n================ FINAL SUMMARY =================")
    print(df.to_string(index=False))

    # OPTIONAL FINAL PLOT (ONLY ONCE)
    # df.groupby("Sigma")["Best PSNR"].mean().plot(kind="bar")
    # plt.show()


if __name__ == "__main__":
    main()