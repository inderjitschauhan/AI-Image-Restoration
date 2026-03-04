import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from math import log10
import pandas as pd

# ================= CONFIG =================
TRAIN_DIR = r"data/train"
WEIGHTS_DIR = r"weights/dncnn"
NOISE_STD = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)


# ================= FLEXIBLE DnCNN =================
class DnCNN(nn.Module):
    def __init__(self, depth=17, channels=64, use_bn=False):
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Conv2d(3, channels, 3, 1, 1))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(channels, channels, 3, 1, 1))
            if use_bn:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        layers.append(nn.Conv2d(channels, 3, 3, 1, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ================= PSNR =================
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * log10(1.0 / np.sqrt(mse))


# ================= RANDOM IMAGE =================
def get_random_image(folder):
    images = [f for f in os.listdir(folder)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if len(images) == 0:
        raise ValueError("No images found in training directory.")

    img_name = random.choice(images)
    img_path = os.path.join(folder, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    return img, img_name


# ================= UNIVERSAL LOADER =================
def load_model_universal(model_path):

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False  # safe for all old checkpoints
    )

    # ===== Case 1: Full model saved =====
    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(DEVICE)
        model.eval()
        return model

    # ===== Case 2: Dict checkpoints =====
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError("Unknown checkpoint format")

    # ===== Fix key prefixes =====
    fixed_state = {}
    for k, v in state.items():

        if k.startswith("model."):
            k = k.replace("model.", "")
        if k.startswith("module."):
            k = k.replace("module.", "")

        # If saved as model.net.state_dict()
        if not k.startswith("net."):
            k = "net." + k

        fixed_state[k] = v

    # ===== Detect channels =====
    first_conv_weight = None
    for k, v in fixed_state.items():
        if "weight" in k and len(v.shape) == 4:
            first_conv_weight = v
            break

    if first_conv_weight is None:
        raise ValueError("Could not detect Conv weights")

    channels = first_conv_weight.shape[0]

    # ===== Detect depth =====
    conv_count = len([
        k for k, v in fixed_state.items()
        if "weight" in k and len(v.shape) == 4
    ])

    depth = conv_count

    # ===== Detect BatchNorm =====
    use_bn = any("running_mean" in k for k in fixed_state.keys())

    print(f"Detected → depth={depth}, channels={channels}, BN={use_bn}")

    # ===== Build model =====
    model = DnCNN(depth=depth, channels=channels, use_bn=use_bn).to(DEVICE)
    model.load_state_dict(fixed_state, strict=False)
    model.eval()

    return model


# ================= MAIN =================
def main():

    print("Using device:", DEVICE)

    clean_img, img_name = get_random_image(TRAIN_DIR)
    print("Testing image:", img_name)

    # Add Gaussian Noise
    noise = np.random.normal(
        0, NOISE_STD / 255.0, clean_img.shape
    ).astype(np.float32)

    noisy_img = np.clip(clean_img + noise, 0., 1.)

    noisy_tensor = torch.from_numpy(noisy_img) \
        .permute(2, 0, 1) \
        .unsqueeze(0) \
        .float() \
        .to(DEVICE)

    results = []

    for weight_file in os.listdir(WEIGHTS_DIR):

        if not weight_file.endswith(".pth"):
            continue

        model_path = os.path.join(WEIGHTS_DIR, weight_file)

        try:
            model = load_model_universal(model_path)

            # Residual inference
            with torch.no_grad():
                predicted_noise = model(noisy_tensor)
                denoised_tensor = noisy_tensor - predicted_noise

            denoised = denoised_tensor.squeeze() \
                .permute(1, 2, 0) \
                .cpu() \
                .numpy()

            denoised = np.clip(denoised, 0., 1.)

            psnr = calculate_psnr(clean_img, denoised)
            results.append([weight_file, round(psnr, 2)])

            print(f"{weight_file} → PSNR: {psnr:.2f} dB\n")

        except Exception as e:
            print(f"Error loading {weight_file}: {e}\n")

    if len(results) == 0:
        print("No models evaluated.")
        return

    df = pd.DataFrame(results, columns=["Model", "PSNR (dB)"])
    df = df.sort_values(by="PSNR (dB)", ascending=False)

    print("\n================ Model Comparison ================")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
