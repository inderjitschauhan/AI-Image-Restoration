import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

# ================= CONFIG =================
TRAIN_DIR = os.path.abspath(r".\data\train")
SAVE_DIR = os.path.abspath(r".\weights\dncnn")
os.makedirs(SAVE_DIR, exist_ok=True)

PATCH_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4

SIGMA = 25   # change to 25 or 50

PATCHES_PER_IMAGE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

best_psnr = 0

# ================= METRICS =================
def psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return 10 * torch.log10(1.0 / mse)

def ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    sigma1 = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

    return (((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) /
           ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))).mean()

# ================= DATASET =================
class DenoiseDataset(Dataset):
    def __init__(self, root, patch_size=64, patches_per_image=2):
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.patch = patch_size
        self.patches_per_image = patches_per_image

    def __len__(self):
        return len(self.files) * self.patches_per_image

    def __getitem__(self, idx):
        file_index = idx % len(self.files)

        img = cv2.imread(self.files[file_index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        H, W, _ = img.shape
        ps = self.patch

        if H < ps or W < ps:
            img = cv2.resize(img, (ps, ps))

        H, W, _ = img.shape

        y = np.random.randint(0, H - ps + 1)
        x = np.random.randint(0, W - ps + 1)

        clean = img[y:y+ps, x:x+ps]
        clean = torch.from_numpy(clean).permute(2, 0, 1).float()

        # Real noise
        shot_noise = torch.poisson(clean * 255.0) / 255.0 - clean
        read_noise = torch.randn_like(clean) * (SIGMA / 255.0)

        noisy = clean + shot_noise + read_noise
        noisy = torch.clamp(noisy, 0.0, 1.0)

        return noisy, clean

# ================= MODEL =================
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

# ================= TRAINING =================
if __name__ == "__main__":

    dataset = DenoiseDataset(TRAIN_DIR, PATCH_SIZE, PATCHES_PER_IMAGE)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = DnCNN(depth=17, channels=64).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Plot setup
    plt.ion()
    fig1 = plt.figure(figsize=(6,4))
    fig2 = plt.figure(figsize=(6,4))

    train_loss_hist = []
    val_psnr_hist = []
    val_ssim_hist = []

    print("Starting Training")
    print("Device:", DEVICE)
    print("Sigma:", SIGMA)

    for epoch in range(1, EPOCHS + 1):

        # TRAIN
        model.train()
        train_loss = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            optimizer.zero_grad()
            pred_noise = model(noisy)

            loss = criterion(pred_noise, noisy - clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATION
        model.eval()
        val_psnr = 0
        val_ssim = 0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(DEVICE)
                clean = clean.to(DEVICE)

                pred_noise = model(noisy)
                denoised = torch.clamp(noisy - pred_noise, 0, 1)

                val_psnr += psnr(denoised, clean).item()
                val_ssim += ssim(denoised, clean).item()

        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        print(f"Epoch {epoch:02d} | Loss: {train_loss:.3e} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

        train_loss_hist.append(train_loss)
        val_psnr_hist.append(val_psnr)
        val_ssim_hist.append(val_ssim)

        # SAVE
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, f"Dncnn_17_Real_Last_{SIGMA}.pth"))

      
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"Dncnn_17_Real_Best_{SIGMA}.pth"))
            print("Best model saved.")

        # LIVE PLOTS
        plt.figure(fig1.number)
        plt.clf()
        plt.plot(train_loss_hist, label="Train Loss")
        plt.xlabel("Epoch")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid()

        plt.figure(fig2.number)
        plt.clf()
        plt.plot(val_psnr_hist, label="PSNR")
        plt.plot(val_ssim_hist, label="SSIM")
        plt.xlabel("Epoch")
        plt.title("Validation Metrics")
        plt.legend()
        plt.grid()

        plt.pause(0.001)

    plt.ioff()
    plt.show()

    print("Training Finished")
    print("Best PSNR:", best_psnr)
