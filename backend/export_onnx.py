"""Convert a U-Net trained with train.py (model.pt) into the .onnx file the desktop app loads.

    python backend/export_onnx.py runs/<run>/model.pt models/cauliflower_unet.onnx

Replace models/cauliflower_unet.onnx to change the built-in model of the next installer build,
or pick the .onnx file in the app (Home tab, "Choose model file...") to use it right away.
The script checks that ONNX Runtime gives the same output as PyTorch.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
import argparse

import numpy as np
import torch

from cauliquant.unet import load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_pt")
    ap.add_argument("out_onnx")
    a = ap.parse_args()
    net = load_model(a.model_pt)
    torch.onnx.export(net, torch.randn(1, 1, 512, 512), a.out_onnx, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}}, opset_version=17,
                      dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(a.out_onnx, providers=["CPUExecutionProvider"])
    for shape in [(512, 512), (448, 576)]:
        x = torch.randn(1, 1, *shape)
        d = np.abs(net(x).detach().numpy() - s.run(None, {"image": x.numpy()})[0]).max()
        print(f"tile {shape}: max difference PyTorch vs ONNX {d:.2e}")
        assert d < 1e-3
    print(f"wrote {a.out_onnx}")


if __name__ == "__main__":
    main()
