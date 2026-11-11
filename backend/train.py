"""Workflow 2: train the U-Net on each raw image in folder A and its marked
version in folder B, and record how the network's parameters change after
every image.

    python train.py --a A_raw --b B_marked [--labels human|all] [--epochs 3]

--labels human (the default) trains only on particles a person classified in
the labeling bench; every other particle is ignored. --labels all also uses
the draft classes from workflow 1, which is only useful for testing the
pipeline: a network trained on drafts learns to copy the draft rule.

Each run writes runs/<date-time>/ with model.pt, param_log.json and
training_monitor.html (open it in any browser).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))  # repo root, for cauliquant
import argparse
import base64
import datetime
import glob
import io
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cauliquant import marks
from cauliquant.unet import UNet, BLOCKS, normalise, save_model
from cauliquant import monitor

PREVIEW_COLOURS = np.array([[40, 44, 52], [47, 127, 168], [217, 119, 31], [110, 110, 110]], np.uint8)


def training_target(b_dir, name, mode):
    """Class per pixel for training; 255 where the network should not learn."""
    rows = marks.read_rows(b_dir, name)
    lab = np.array(Image.open(marks.paths(b_dir, name)["objects"])).astype(np.int32)
    lut = np.full(lab.max() + 1, 255, np.uint8)
    lut[0] = marks.BG
    keep = {"human"} if mode == "human" else {"human", "draft", "model"}
    n = 0
    for r in rows:
        if r["source"] in keep and r["class"] in ("dense", "cauliflower"):
            lut[int(r["id"])] = marks.CODE[r["class"]]
            n += 1
    t = lut[lab]
    t[(lab == 0) & (np.array(Image.open(marks.paths(b_dir, name)["mask"])) == marks.IGNORE)] = 255   # text overlay band
    if mode == "human" and n == 0:
        return None, 0
    if mode == "human":
        # background is only trusted near labeled particles
        from scipy import ndimage as ndi
        near = ndi.binary_dilation(np.isin(t, (1, 2)), iterations=40)
        t[(t == 0) & ~near] = 255
    return t, n


def load_dataset(a_dir, b_dir, mode):
    items = []
    for f in sorted(glob.glob(os.path.join(a_dir, "*.tif"))):
        name = os.path.splitext(os.path.basename(f))[0]
        if marks.read_rows(b_dir, name) is None:
            continue
        from cauliquant.segment import load_sem
        gray, _ = load_sem(f)
        t, n = training_target(b_dir, name, mode)
        if t is None:
            continue
        items.append(dict(name=name, x=normalise(gray), t=t, n=n))
    return items


def sample_tiles(item, k, size, rng):
    """k random tiles, most of them centred on a labeled particle pixel."""
    H, W = item["t"].shape
    fg = np.argwhere(np.isin(item["t"], (1, 2)))
    xs, ts = [], []
    for j in range(k):
        if len(fg) and rng.random() < 0.8:
            cy, cx = fg[rng.integers(len(fg))]
        else:
            cy, cx = rng.integers(H), rng.integers(W)
        y0 = int(np.clip(cy - size // 2, 0, H - size)); x0 = int(np.clip(cx - size // 2, 0, W - size))
        x = item["x"][y0:y0 + size, x0:x0 + size]; t = item["t"][y0:y0 + size, x0:x0 + size]
        r = rng.integers(4)
        x, t = np.rot90(x, r), np.rot90(t, r)
        if rng.random() < 0.5:
            x, t = x[:, ::-1], t[:, ::-1]
        xs.append(x.copy()); ts.append(t.copy())
    return torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ts)).long()


def class_weights(items):
    c = np.zeros(3)
    for it in items:
        c += np.bincount(it["t"][it["t"] < 3].ravel(), minlength=3)[:3]
    w = 1.0 / np.sqrt(np.maximum(c, 1))
    return torch.tensor(w / w.sum() * 3, dtype=torch.float32)


def iou(pred, t, k):
    v = t != 255
    inter = ((pred == k) & (t == k) & v).sum(); union = (((pred == k) | (t == k)) & v).sum()
    return float(inter / union) if union else None


class ParamRecorder:
    """Snapshots of the weights, taken after each image."""

    def __init__(self, net, bins=41):
        self.net = net
        self.names = [n for n, p in net.named_parameters() if p.dim() > 1]
        self.block = {n: net.block_of(n) for n in self.names}
        self.w0 = {n: p.detach().clone() for n, p in net.named_parameters() if n in self.names}
        self.prev = {n: w.clone() for n, w in self.w0.items()}
        self.grad_sq = {b: 0.0 for b in BLOCKS}
        self.grad_n = 0
        self.edges = {}
        for b in BLOCKS:
            v = torch.cat([self.w0[n].flatten() for n in self.names if self.block[n] == b])
            s = float(v.std()) * 4
            self.edges[b] = np.linspace(-s, s, bins + 1)
        self.n_params = sum(p.numel() for p in net.parameters())

    def after_backward(self):
        for n, p in self.net.named_parameters():
            if n in self.block and p.grad is not None:
                self.grad_sq[self.block[n]] += float(p.grad.pow(2).sum())
        self.grad_n += 1

    def snapshot(self):
        cur = {n: p.detach() for n, p in self.net.named_parameters() if n in self.block}
        out = {}
        for b in BLOCKS:
            ns = [n for n in self.names if self.block[n] == b]
            w = torch.cat([cur[n].flatten() for n in ns]); w0 = torch.cat([self.w0[n].flatten() for n in ns])
            wp = torch.cat([self.prev[n].flatten() for n in ns])
            hist, _ = np.histogram(w.numpy(), bins=self.edges[b])
            out[b] = dict(norm=float(w.norm()), drift=float((w - w0).norm() / w0.norm()),
                          step=float((w - wp).norm() / wp.norm()),
                          grad=(self.grad_sq[b] / max(self.grad_n, 1)) ** 0.5,
                          hist=hist.tolist())
        self.prev = {n: w.clone() for n, w in cur.items()}
        self.grad_sq = {b: 0.0 for b in BLOCKS}; self.grad_n = 0
        first = cur[[n for n in self.names if self.block[n] == "enc1"][0]][:, 0]   # (16, 3, 3)
        head = cur["head.weight"][:, :, 0, 0]                                           # (3, 16)
        return out, first.numpy().round(4).tolist(), head.numpy().round(4).tolist()


def preview_png(net, item, box):
    y0, x0, s = box
    with torch.no_grad():
        net.eval()
        x = torch.from_numpy(item["x"][y0:y0 + s, x0:x0 + s].copy())[None, None]
        pred = net(x).argmax(1)[0].numpy()
        net.train()
    img = Image.fromarray(PREVIEW_COLOURS[pred]).resize((s // 2, s // 2), Image.NEAREST)
    buf = io.BytesIO(); img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def still_png(arr, colours=None):
    img = Image.fromarray(arr if colours is None else colours[np.where(arr == 255, 3, arr)])
    s = arr.shape[0] // 2
    img = img.resize((s, s), Image.BILINEAR if colours is None else Image.NEAREST)
    buf = io.BytesIO(); img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--labels", choices=["human", "all"], default="human")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8, help="training steps per image")
    ap.add_argument("--batch", type=int, default=6); ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    items = load_dataset(args.a, args.b, args.labels)
    if not items:
        raise SystemExit("No A/B pairs with usable labels. Run classify_folder.py and import_labels.py first.")
    print(f"{len(items)} image pairs, labels: {args.labels}")
    run_dir = os.path.join(args.runs, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    net = UNet(); net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(items)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total * args.steps)
    cw = class_weights(items)
    rec = ParamRecorder(net)

    # fixed preview: the busiest 384 px square of the last image
    pv = items[-1]; s = 384
    H, W = pv["t"].shape
    best, box = -1, (0, 0, s)
    for y0 in range(0, H - s, 96):
        for x0 in range(0, W - s, 96):
            c = np.isin(pv["t"][y0:y0 + s, x0:x0 + s], (1, 2)).sum()
            if c > best:
                best, box = c, (y0, x0, s)
    raw = ((pv["x"][box[0]:box[0] + s, box[1]:box[1] + s] - pv["x"].min()) / (np.ptp(pv["x"]) + 1e-6) * 255).astype(np.uint8)

    first, head = rec.snapshot()[1:]
    log = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), labels=args.labels,
               settings=vars(args), n_params=rec.n_params, blocks=BLOCKS,
               hist_edges={b: [round(float(e), 5) for e in rec.edges[b]] for b in BLOCKS},
               images=[dict(name=it["name"], particles=it["n"]) for it in items],
               preview=dict(image=pv["name"], box=box, raw=still_png(raw), target=still_png(pv["t"][box[0]:box[0] + s, box[1]:box[1] + s], PREVIEW_COLOURS)),
               start=dict(filters=first, head=head, preview=preview_png(net, pv, box)),
               snapshots=[])
    t0 = time.time()
    k = 0
    for ep in range(args.epochs):
        order = list(range(len(items))) if ep == 0 else list(rng.permutation(len(items)))
        for idx in order:
            it = items[idx]
            # how well the network does on this image before it learns from it
            net.eval()
            with torch.no_grad():
                xb, tb = sample_tiles(it, 6, args.tile, np.random.default_rng(1000 + idx))
                out = net(xb); before = float(F.cross_entropy(out, tb, weight=cw, ignore_index=255))
                pred = out.argmax(1).numpy(); tn = tb.numpy()
                acc = float(((pred == tn) & (tn != 255)).sum() / max((tn != 255).sum(), 1))
                iou_c = iou(pred, tn, 2)
            net.train()
            losses = []
            for _ in range(args.steps):
                xb, tb = sample_tiles(it, args.batch, args.tile, rng)
                loss = F.cross_entropy(net(xb), tb, weight=cw, ignore_index=255)
                opt.zero_grad(); loss.backward(); rec.after_backward(); opt.step(); sched.step()
                losses.append(float(loss))
            k += 1
            blocks, filters, headw = rec.snapshot()
            log["snapshots"].append(dict(k=k, epoch=ep + 1, image=it["name"], lr=sched.get_last_lr()[0],
                                         loss=float(np.mean(losses)), loss_before=before, acc_before=acc,
                                         iou_cauli_before=iou_c, blocks=blocks, filters=filters, head=headw,
                                         preview=preview_png(net, pv, box)))
            print(f"[{k:3d}/{total}] epoch {ep + 1}  {it['name']:28s} loss {np.mean(losses):.3f}  before {before:.3f}  {time.time() - t0:5.0f}s", flush=True)

    save_model(net, os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "param_log.json"), "w") as fh:
        json.dump(log, fh)
    html = monitor.build(log)
    with open(os.path.join(run_dir, "training_monitor.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", run_dir)


if __name__ == "__main__":
    main()
