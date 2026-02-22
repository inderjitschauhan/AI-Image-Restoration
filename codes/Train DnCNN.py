import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

# ================= CONFIG =================
TRAIN_DIR = os.path.abspath(r".\data\train")
SAVE_DIR = os.path.abspath(r".\weights\dncnn")
os.makedirs(SAVE_DIR, exist_ok=True)
#print("Train Dir:", TRAIN_DIR)
#print("Save Dir:", SAVE_DIR)

os.makedirs(SAVE_DIR, exist_ok=True)

PATCH_SIZE = 64
BATCH_SIZE = 16   # large batch; adjust to GPU memory
EPOCHS = 50
LR = 1e-3
SIGMA = 25
PATCHES_PER_IMAGE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

best_psnr = 0

# ================= PSNR =================
def psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return 10 * torch.log10(1.0 / mse)

# ================= DATASET =================
class DenoiseDataset(Dataset):
    def __init__(self, root, patch_size=64, patches_per_image=50):
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
        noise = np.random.normal(0, SIGMA/255.0, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise, 0, 1)

        clean = torch.from_numpy(clean).permute(2,0,1)
        noisy = torch.from_numpy(noisy).permute(2,0,1)

        return noisy, clean

# ================= MODEL =================
class DnCNN(nn.Module):
    def __init__(self, depth=8, channels=32):
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

# ================= MODEL DEBUG FUNCTION =================
def print_model_details(model, device, input_size=(1, 3, 64, 64)):
    print("\n================ MODEL SUMMARY ================\n")
    
    # Full architecture
    print("Full Architecture:\n")
    print(model)
    
    print("\n---------------- Layer-wise Output Shapes ----------------\n")
    
    x = torch.randn(input_size).to(device)
    
    for i, layer in enumerate(model.net):
        x = layer(x)
        print(f"Layer {i:02d} | {layer.__class__.__name__:<15} | Output Shape: {tuple(x.shape)}")
    
    print("\n---------------- Parameter Details ----------------\n")
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"{name:<30} Shape: {tuple(param.shape)} | Params: {num_params}")
    
    print("\n--------------------------------------------------")
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print("==================================================\n")

# ================= MAIN TRAINING =================
if __name__ == "__main__":
    
    dataset = DenoiseDataset(TRAIN_DIR, PATCH_SIZE, PATCHES_PER_IMAGE)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # safe for Windows
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = DnCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    plt.ion()
    fig1 = plt.figure(figsize=(7,5))
    fig2 = plt.figure(figsize=(7,5))

    train_loss_hist = []
    val_loss_hist = []
    val_psnr_hist = []

    print("Starting Training")
    print("Device:", DEVICE, "| Images:", len(dataset))

    for epoch in range(1, EPOCHS+1):

        model.train()
        train_loss = 0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

            optimizer.zero_grad()
            pred_noise = model(noisy)
            loss = criterion(pred_noise, noisy - clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_psnr = 0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

                pred_noise = model(noisy)
                loss = criterion(pred_noise, noisy - clean)
                val_loss += loss.item()

                denoised = torch.clamp(noisy - pred_noise, 0, 1)
                val_psnr += psnr(denoised, clean).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.3e} | "
              f"Val Loss: {val_loss:.3e} | Val PSNR: {val_psnr:.2f} dB")

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_psnr_hist.append(val_psnr)

        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "model_last.pth"))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "model_best.pth"))
            print("Best model saved.")

        plt.figure(fig1.number)
        plt.clf()
        plt.plot(train_loss_hist, label="Train Loss")
        plt.plot(val_loss_hist, label="Val Loss")
        plt.xlabel("Epoch")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)

        plt.figure(fig2.number)
        plt.clf()
        plt.plot(val_psnr_hist, label="Val PSNR")
        plt.xlabel("Epoch")
        plt.title("PSNR Curve")
        plt.legend()
        plt.grid(True)

        plt.pause(0.001)

    plt.ioff()
    plt.show()

    # Optional: print model details at the end
    print_model_details(model, DEVICE)