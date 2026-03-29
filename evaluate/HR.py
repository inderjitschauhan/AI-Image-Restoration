import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim

# ================= FLAGS =================
USE_NOISE = False
USE_DNCNN = False
USE_SWIN = True
SCALE = 4

# ================= CONFIG =================
TRAIN_DIR = r"data"
DNCNN_DIR = r"weights/dncnn"
SWIN_PATH = r"weights/SR/best_mini_swinir.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= PREPROCESS =================
def mod_crop(img, scale):
    h, w = img.shape[:2]
    h = h - (h % scale)
    w = w - (w % scale)
    return img[:h, :w]

def downscale_hr(img, scale=4):
    h, w = img.shape[:2]
    return cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)

# ================= IMAGE =================
def get_random_image(folder):
    valid_images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                valid_images.append(os.path.join(root, f))

    path = random.choice(valid_images)
    print("Selected:", path)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

# ================= NOISE =================
def add_real_noise(img):
    noise = np.random.normal(0, 0.05, img.shape)
    return np.clip(img + noise, 0., 1.).astype(np.float32)

# ================= METRICS =================
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 100 if mse == 0 else 20 * log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2, data_range=1.0)

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

# ================= SWIN =================
class MiniSwinIR(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.conv_first = nn.Conv2d(3, dim, 3, 1, 1)
        self.conv_mid = nn.Conv2d(dim, dim, 3,1,1)
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim*(4**2),3,1,1),
            nn.PixelShuffle(4),
            nn.Conv2d(dim,3,3,1,1)
        )

    def forward(self,x):
        x = self.conv_first(x)
        res = x
        x = self.conv_mid(x)
        x = x + res
        return self.upsample(x)

def load_swin_model(path):
    model = MiniSwinIR().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    return model.eval()

# ================= MAIN =================
def main():

    # -------- LOAD HR --------
    hr = get_random_image(TRAIN_DIR)
    hr = mod_crop(hr, SCALE)

    # -------- CREATE LR --------
    lr = downscale_hr(hr, SCALE)

    print("HR:", hr.shape, "LR:", lr.shape)

    # -------- NOISE --------
    if USE_NOISE:
        noisy = add_real_noise(lr)
    else:
        noisy = lr.copy()

    print("Noise diff:", np.mean(np.abs(lr - noisy)))

    noisy_tensor = torch.from_numpy(noisy).permute(2,0,1).unsqueeze(0).to(DEVICE)

    # -------- DNCNN --------
    best_dncnn = noisy.copy()

    if USE_DNCNN:
        best_psnr = -1

        for f in os.listdir(DNCNN_DIR):
            if f.endswith(".pth"):
                model = load_dncnn(os.path.join(DNCNN_DIR, f))

                with torch.no_grad():
                    out = noisy_tensor - model(noisy_tensor)

                img = np.clip(out.squeeze().permute(1,2,0).cpu().numpy(),0,1)

                psnr = calculate_psnr(hr, cv2.resize(img, (hr.shape[1], hr.shape[0])))
                print(f"{f} → {psnr:.2f}")

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_dncnn = img

        print("Best DnCNN:", best_psnr)

    # -------- SWIN --------
    if USE_SWIN:
        swin = load_swin_model(SWIN_PATH)
        inp = torch.from_numpy(best_dncnn).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            final = swin(inp)

        final = np.clip(final.squeeze().permute(1,2,0).cpu().numpy(),0,1)
    else:
        final = best_dncnn.copy()

    # -------- METRICS --------
    final_resized = cv2.resize(final, (hr.shape[1], hr.shape[0]))

    print("\n=========== RESULTS ===========")
    print("Noisy :", calculate_psnr(hr, cv2.resize(noisy, (hr.shape[1], hr.shape[0]))))
    print("DnCNN :", calculate_psnr(hr, cv2.resize(best_dncnn, (hr.shape[1], hr.shape[0]))))
    print("Final :", calculate_psnr(hr, final_resized))


if __name__ == "__main__":
    main()