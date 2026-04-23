import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= CONFIG =================
DNCNN_DIR = r"weights/dncnn"
SWIN_DN_PATH = r"weights/sr/best_swinir_dn.pth"
SWIN_SR_PATH = r"weights/sr/best_swinir_sr.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 4

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

# ================= SWIN BLOCK =================
class SwinBlock(nn.Module):
    def __init__(self, dim, heads=2, window_size=8, shift=False, mlp_ratio=4):
        super().__init__()

        self.window_size = window_size
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return x  # only for architecture printing

# ================= SWINIR DN =================
class SwinIR_DN(nn.Module):
    def __init__(self, dim=48):
        super().__init__()

        self.conv_first = nn.Conv2d(3, dim, 3, 1, 1)

        self.blocks = nn.Sequential(
            SwinBlock(dim, 2, 8, False, mlp_ratio=2),
            SwinBlock(dim, 2, 8, True,  mlp_ratio=2),
            SwinBlock(dim, 2, 8, False, mlp_ratio=2),
            SwinBlock(dim, 2, 8, True,  mlp_ratio=2),
        )

        self.conv_mid = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(dim, 3, 3, 1, 1)

    def forward(self, x):
        return x

# ================= SWINIR SR =================
class MiniSwinIR(nn.Module):
    def __init__(self, dim=96):
        super().__init__()

        self.conv_first = nn.Conv2d(3, dim, 3, 1, 1)

        self.blocks = nn.Sequential(
            SwinBlock(dim, 4, 8, False, mlp_ratio=4),
            SwinBlock(dim, 4, 8, True,  mlp_ratio=4),
            SwinBlock(dim, 4, 8, False, mlp_ratio=4),
            SwinBlock(dim, 4, 8, True,  mlp_ratio=4),
        )

        self.conv_mid = nn.Conv2d(dim, dim, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * (SCALE ** 2), 3, 1, 1),
            nn.PixelShuffle(SCALE),
            nn.Conv2d(dim, 3, 3, 1, 1)
        )

    def forward(self, x):
        return x

# ================= UTIL =================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model, name):
    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"{'='*60}")

    print("\n🔹 Architecture:\n")
    print(model)

    total_params = count_parameters(model)
    print(f"\n🔹 Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    print("\n🔹 Layer-wise Details:\n")
    for lname, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            print(f"{lname:30s} | {module.__class__.__name__:20s} | {params:,}")

# ================= LOAD DNCNN =================
def load_dncnn(path):
    state = torch.load(path, map_location=DEVICE,weights_only=True)
    state = state.get("model_state_dict", state)

    convs = [v for v in state.values() if len(v.shape) == 4]
    depth = len(convs)
    channels = convs[0].shape[0]
    use_bn = any("running_mean" in k for k in state.keys())

    model = DnCNN(depth, channels, use_bn).to(DEVICE)
    model.load_state_dict(state, strict=False)

    return model, depth

# ================= MAIN =================
def main():
    printed = set()

    # ===== DNCNN =====
    print("\n========= DNCNN MODELS =========")

    for f in os.listdir(DNCNN_DIR):
        if not f.endswith(".pth"):
            continue

        path = os.path.join(DNCNN_DIR, f)
        model, depth = load_dncnn(path)

        key = f"DnCNN-{depth}"

        if key in printed:
            continue

        print_model_summary(model, key)
        printed.add(key)

    # ===== SWINIR DN =====
    print("\n========= SWINIR DN =========")

    swin_dn = SwinIR_DN().to(DEVICE)

    state = torch.load(SWIN_DN_PATH, map_location=DEVICE,weights_only=True)
    state = state.get("model_state_dict", state)

    swin_dn.load_state_dict(state, strict=False)

    print_model_summary(swin_dn, "SwinIR-DN")

    # ===== SWINIR SR =====
    print("\n========= SWINIR SR =========")

    swin_sr = MiniSwinIR().to(DEVICE)

    state = torch.load(SWIN_SR_PATH, map_location=DEVICE,weights_only=True)
    state = state.get("model_state_dict", state)

    swin_sr.load_state_dict(state, strict=False)

    print_model_summary(swin_sr, "SwinIR-SR")

# ================= RUN =================
if __name__ == "__main__":
    main()