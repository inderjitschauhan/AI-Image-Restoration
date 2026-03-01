import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from math import log10
import pandas as pd

# ================= CONFIG =================
TRAIN_DIR = r"data/set5"
WEIGHTS_DIR = r"weights/dncnn"
NOISE_STD = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)


# ================= DnCNN =================
class DnCNN(nn.Module):
    def __init__(self, depth, channels):
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

    img_name = random.choice(images)
    img_path = os.path.join(folder, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    return img, img_name


# ================= MAIN =================
def main():
    print("Using device:", DEVICE)

    clean_img, img_name = get_random_image(TRAIN_DIR)
    print("Testing image:", img_name)

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
            state = torch.load(model_path, map_location=DEVICE, weights_only=True)

            # -------- Fix key prefix --------
            new_state = {}
            for k, v in state.items():
                if k.startswith("model."):
                    new_key = k.replace("model.", "net.")
                else:
                    new_key = k
                new_state[new_key] = v

            # -------- Auto-detect channels --------
            first_weight = new_state["net.0.weight"]
            channels = first_weight.shape[0]

            # -------- Auto-detect depth --------
            conv_layers = len([k for k in new_state.keys() if "weight" in k])
            depth = conv_layers  # one conv per weight

            print(f"{weight_file} detected → depth: {depth}, channels: {channels}")

            # Build correct model
            model = DnCNN(depth=depth, channels=channels).to(DEVICE)
            model.load_state_dict(new_state)
            model.eval()

            # -------- Residual inference --------
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