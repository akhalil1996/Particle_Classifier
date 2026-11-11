"""A small U-Net: grey SEM tile in, three class scores per pixel out
(background, dense, cauliflower)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCKS = ["enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1", "head"]


def double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=16, n_classes=3):
        super().__init__()
        b = base
        self.enc1, self.enc2, self.enc3 = double_conv(1, b), double_conv(b, 2 * b), double_conv(2 * b, 4 * b)
        self.bottleneck = double_conv(4 * b, 8 * b)
        self.up3, self.dec3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), double_conv(8 * b, 4 * b)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), double_conv(4 * b, 2 * b)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * b, b, 2, 2), double_conv(2 * b, b)
        self.head = nn.Conv2d(b, n_classes, 1)
        self.config = dict(base=base, n_classes=n_classes)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        m = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(m), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)

    def block_of(self, param_name):
        """Group each weight tensor under one of the eight blocks above."""
        top = param_name.split(".")[0]
        return {"up3": "dec3", "up2": "dec2", "up1": "dec1"}.get(top, top)


def normalise(gray):
    g = gray.astype(np.float32) / 255.0
    return (g - g.mean()) / (g.std() + 1e-6)


def save_model(net, path):
    torch.save({"config": net.config, "state": net.state_dict()}, path)


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    net = UNet(**ck["config"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net


@torch.no_grad()
def predict_probs(net, gray, tile=512, margin=32):
    """Class probabilities for a whole image, computed tile by tile."""
    net.eval()
    x = normalise(gray)
    H, W = x.shape
    out = np.zeros((3, H, W), np.float32)
    step = tile - 2 * margin
    for y in range(0, H, step):
        for x0 in range(0, W, step):
            ys, xs = max(0, y - margin), max(0, x0 - margin)
            ye, xe = min(H, y + step + margin), min(W, x0 + step + margin)
            t = x[ys:ye, xs:xe]
            ph, pw = (-t.shape[0]) % 8, (-t.shape[1]) % 8
            t = np.pad(t, ((0, ph), (0, pw)), mode="reflect")
            p = torch.softmax(net(torch.from_numpy(t)[None, None]), 1)[0].numpy()
            p = p[:, : ye - ys, : xe - xs]
            yy, xx = y - ys, x0 - xs
            out[:, y: min(H, y + step), x0: min(W, x0 + step)] = p[:, yy: yy + min(step, H - y), xx: xx + min(step, W - x0)]
    return out
