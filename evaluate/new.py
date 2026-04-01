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
DATA_DIR = r"data"
DNCNN_DIR = r"weights/dncnn"
SWIN_DN_PATH = r"weights/SR/best_swinir_dn.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= DNCNN =================
class DnCNN(nn.Module):
    def __init__(self, depth=17, channels=64, use_bn=False):
        super().__init__()
        layers = [nn.Conv2d(3, channels, 3,1,1), nn.ReLU(inplace=True)]

        for _ in range(depth-2):
            layers.append(nn.Conv2d(channels, channels,3,1,1))
            if use_bn:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(channels,3,3,1,1))
        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)

# ================= SWIN DN =================
class SwinBlock(nn.Module):
    def __init__(self, dim, shift=False):
        super().__init__()
        self.ws = 8
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 2, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim)
        )

    def forward(self,x):
        B,C,H,W = x.shape
        ws = self.ws

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x,(0,pad_w,0,pad_h))
        _,_,Hp,Wp = x.shape

        if self.shift:
            x = torch.roll(x,(-ws//2,-ws//2),(2,3))

        x = x.view(B,C,Hp//ws,ws,Wp//ws,ws)
        x = x.permute(0,2,4,3,5,1).contiguous().view(-1,ws*ws,C)

        x = self.norm1(x)
        attn,_ = self.attn(x,x,x)
        x = x + attn
        x = x + self.mlp(self.norm2(x))

        x = x.view(B,Hp//ws,Wp//ws,ws,ws,-1)
        x = x.permute(0,5,1,3,2,4).contiguous().view(B,-1,Hp,Wp)

        if self.shift:
            x = torch.roll(x,(ws//2,ws//2),(2,3))

        return x[:,:,:H,:W]

class SwinIR_DN(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.conv_first = nn.Conv2d(3, dim,3,1,1)

        self.blocks = nn.Sequential(
            SwinBlock(dim, False),
            SwinBlock(dim, True),
            SwinBlock(dim, False),
            SwinBlock(dim, True),
        )

        self.conv_mid = nn.Conv2d(dim,dim,3,1,1)
        self.conv_last = nn.Conv2d(dim,3,3,1,1)

    def forward(self,x):
        x = self.conv_first(x)
        res = x
        x = self.blocks(x)
        x = self.conv_mid(x)
        x = x + res
        return self.conv_last(x)

# ================= METRICS =================
def psnr(a,b):
    mse = np.mean((a-b)**2)
    return 100 if mse==0 else 20*np.log10(1.0/np.sqrt(mse))

def ssim_calc(a,b):
    return ssim(a,b,channel_axis=2,data_range=1.0)

# ================= IMAGE =================
def get_random_image(folder):
    imgs = []
    for root,_,files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                imgs.append(os.path.join(root,f))

    path = random.choice(imgs)
    print("Selected:", path)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)/255.0

# ================= NOISE =================
def add_noise(img):
    std = np.random.uniform(0.09,0.01)
    noise = np.random.randn(*img.shape)*std
    return np.clip(img+noise,0,1)

# ================= LOAD DNCNN =================
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

# ================= MAIN =================
def main():

    NUM_SAMPLES = 3

    # Load models
    dncnn_models = [(f, load_dncnn(os.path.join(DNCNN_DIR,f)))
                    for f in os.listdir(DNCNN_DIR) if f.endswith(".pth")]

    swin = SwinIR_DN().to(DEVICE)
    swin.load_state_dict(torch.load(SWIN_DN_PATH, map_location=DEVICE))
    swin.eval()

    for i in range(NUM_SAMPLES):

        print(f"\n===== SAMPLE {i+1} =====")

        clean = get_random_image(DATA_DIR)
        noisy = add_noise(clean)

        inp = torch.from_numpy(noisy).permute(2,0,1).unsqueeze(0).float().to(DEVICE)


        print(f"Noisy → PSNR: {psnr(clean,noisy):.2f} | SSIM: {ssim_calc(clean,noisy):.4f}")

        # ---- DNCNN ----
        for name, model in dncnn_models:
            with torch.no_grad():
                out = inp - model(inp)

            out = out.squeeze().permute(1,2,0).cpu().numpy()
            out = np.clip(out,0,1)

            print(f"{name} → {psnr(clean,out):.2f} | {ssim_calc(clean,out):.4f}")

        # ---- SWIN ----
        with torch.no_grad():
            out = inp - swin(inp)

        out = out.squeeze().permute(1,2,0).cpu().numpy()
        out = np.clip(out,0,1)

        print(f"SwinIR_DN → {psnr(clean,out):.2f} | {ssim_calc(clean,out):.4f}")

        # ---- SHOW ----
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.imshow(noisy)
        plt.title("Noisy")

        plt.subplot(1,2,2)
        plt.imshow(out)
        plt.title("Swin Output")

        plt.show()

if __name__ == "__main__":
    main()
