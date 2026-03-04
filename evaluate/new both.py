
import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ====================================
# CONFIG
# ====================================
DATA_DIR = r"C:\Users\inder\Documents\AI_Image_Restoration\data\RealSR(V3)\Canon\Train\4"
SCALE = 4
DEVICE = torch.device("cpu")

results = []

# Get all LR4 images
image_list = [f for f in os.listdir(DATA_DIR) if "_LR4" in f]

# Pick random 10
random_images = random.sample(image_list, min(10, len(image_list)))

print("Selected Images:")
for name in random_images:
    print(" -", name)

print("\nProcessing...\n")

for img_name in random_images:

    lr_path = os.path.join(DATA_DIR, img_name)
    hr_path = lr_path.replace("_LR4", "_HR")

    if not os.path.exists(hr_path):
        print("HR not found for:", img_name)
        continue

    # ===============================
    # LOAD LR
    # ===============================
    lr = cv2.imread(lr_path)
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    lr = lr.astype(np.float32) / 255.0

    lr_tensor = torch.from_numpy(lr).permute(2,0,1).unsqueeze(0).to(DEVICE)

    # ===============================
    # DENOISING
    # ===============================
    with torch.no_grad():
        noise_pred = dncnn(lr_tensor)
        denoised_tensor = lr_tensor - noise_pred

    denoised_tensor = denoised_tensor.clamp(0,1)

    # ===============================
    # SUPER RESOLUTION
    # ===============================
    with torch.no_grad():
        sr_tensor = swinir(denoised_tensor)

    sr_tensor = sr_tensor.clamp(0,1)
    sr = sr_tensor.squeeze(0).permute(1,2,0).cpu().numpy()

    # ===============================
    # LOAD HR
    # ===============================
    hr = cv2.imread(hr_path)
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    hr = hr.astype(np.float32) / 255.0

    # ===============================
    # BICUBIC
    # ===============================
    h_lr, w_lr, _ = lr.shape
    bicubic = cv2.resize(lr, (w_lr*SCALE, h_lr*SCALE),
                         interpolation=cv2.INTER_CUBIC)

    # ===============================
    # METRICS
    # ===============================
    psnr_bic = peak_signal_noise_ratio(hr, bicubic, data_range=1.0)
    ssim_bic = structural_similarity(hr, bicubic,
                                     channel_axis=2,
                                     data_range=1.0)

    psnr_sr = peak_signal_noise_ratio(hr, sr, data_range=1.0)
    ssim_sr = structural_similarity(hr, sr,
                                    channel_axis=2,
                                    data_range=1.0)

    results.append([
        img_name,
        psnr_bic,
        ssim_bic,
        psnr_sr,
        ssim_sr
    ])

    print(f"{img_name} done.")

# ====================================
# CREATE TABLE
# ====================================
columns = [
    "Image",
    "PSNR_Bicubic",
    "SSIM_Bicubic",
    "PSNR_DnCNN_SwinIR",
    "SSIM_DnCNN_SwinIR"
]

df = pd.DataFrame(results, columns=columns)

# Add average row
avg_values = df.iloc[:,1:].mean()
df.loc[len(df)] = ["Average"] + list(avg_values)

print("\n==============================")
print(df)
print("==============================")

# Optional save
df.to_csv("RealSR_random10_results.csv", index=False)
print("\nSaved as RealSR_random10_results.csv")