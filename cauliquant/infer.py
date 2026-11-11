"""U-Net inference with ONNX Runtime, so the installed app does not need PyTorch.

A model trained with train.py (model.pt) is converted once with
backend/export_onnx.py; the app then loads the .onnx file.
"""
import numpy as np

from .unet_np import normalise


class UNetONNX:
    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.path = path
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def _logits(self, t):
        return self.session.run(None, {self.input: t[None, None].astype(np.float32)})[0][0]

    def predict_probs(self, gray, tile=512, margin=32, progress=None):
        """Class probabilities (3, H, W): background, dense, cauliflower; computed tile by tile."""
        x = normalise(gray)
        H, W = x.shape
        out = np.zeros((3, H, W), np.float32)
        step = tile - 2 * margin
        ys_all = list(range(0, H, step))
        xs_all = list(range(0, W, step))
        n, k = len(ys_all) * len(xs_all), 0
        for y in ys_all:
            for x0 in xs_all:
                ys, xs = max(0, y - margin), max(0, x0 - margin)
                ye, xe = min(H, y + step + margin), min(W, x0 + step + margin)
                t = x[ys:ye, xs:xe]
                ph, pw = (-t.shape[0]) % 8, (-t.shape[1]) % 8
                t = np.pad(t, ((0, ph), (0, pw)), mode="reflect")
                lg = self._logits(t)
                lg = lg - lg.max(0, keepdims=True)
                p = np.exp(lg)
                p /= p.sum(0, keepdims=True)
                p = p[:, : ye - ys, : xe - xs]
                yy, xx = y - ys, x0 - xs
                out[:, y: min(H, y + step), x0: min(W, x0 + step)] = \
                    p[:, yy: yy + min(step, H - y), xx: xx + min(step, W - x0)]
                k += 1
                if progress:
                    progress(k / n)
        return out
