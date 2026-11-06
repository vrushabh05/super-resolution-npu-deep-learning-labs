"""Export the best checkpoint to a single-file ONNX (opset 11, fixed 1x3x256x256) and compare with PyTorch."""
import sys

import numpy as np
import onnx
import torch

from model import DenoiseSRNet

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pth"
out_path = sys.argv[2] if len(sys.argv) > 2 else "sr_model.onnx"

ckpt = torch.load(ckpt_path, map_location="cpu")
model = DenoiseSRNet(96, 16)
model.load_state_dict(ckpt["model"])
model.eval()

dummy = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy, out_path, export_params=True, opset_version=11, do_constant_folding=True,
                  input_names=["input"], output_names=["output"], dynamic_axes=None)
proto = onnx.load(out_path, load_external_data=True)
onnx.save_model(proto, out_path, save_as_external_data=False)  # one self-contained file
onnx.checker.check_model(onnx.load(out_path))

import onnxruntime as ort
sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
with torch.no_grad():
    ref = model(dummy).clamp(0, 1).numpy()
got = np.clip(sess.run(["output"], {"input": dummy.numpy()})[0], 0, 1)
print("max abs diff PyTorch vs ONNX Runtime:", float(np.abs(ref - got).max()))
