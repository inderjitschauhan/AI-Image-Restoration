import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# LOAD DNCNN
# ===============================
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=17):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(channels, 64, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers-2):
            layers.append(nn.Conv2d(64, 64, 3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, 3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise


dncnn = DnCNN().to(DEVICE)
dncnn.load_state_dict(torch.load("weights/dncnn/dncnn_trained_best50.pth", map_location=DEVICE))
dncnn.eval()

# ===============================
# LOAD SR MODEL (MiniSwinIR)
# ===============================
# (Use same MiniSwinIR class from your training)
sr_model = MiniSwinIR().to(DEVICE)
sr_model.load_state_dict(torch.load("best_mini_swinir.pth", map_location=DEVICE))
sr_model.eval()

# ===============================
# LOAD IMAGE
# ===============================
image_path = "test_LR4.png"
lr_image = Image.open(image_path).convert("RGB")

transform = transforms.ToTensor()
lr_tensor = transform(lr_image).unsqueeze(0).to(DEVICE)

# ===============================
# PIPELINE
# ===============================
with torch.no_grad():

    # Step 1: Denoise
    clean_lr = dncnn(lr_tensor)

    # Step 2: Super Resolution
    sr_tensor = sr_model(clean_lr)

sr_tensor = sr_tensor.clamp(0,1)

# ===============================
# DISPLAY RESULTS
# ===============================
clean_lr_img = transforms.ToPILImage()(clean_lr.squeeze(0).cpu())
sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Noisy LR")
plt.imshow(lr_image)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Denoised LR")
plt.imshow(clean_lr_img)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Final Super Resolved")
plt.imshow(sr_img)
plt.axis("off")

plt.tight_layout()
plt.show()