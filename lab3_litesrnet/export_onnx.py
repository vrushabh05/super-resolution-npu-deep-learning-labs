"""Export the best LiteSRNet checkpoint (EMA weights) to a single-file ONNX model, opset 11.

    python export_onnx.py --checkpoint checkpoints/best_model.pth --out sr_model.onnx
"""
import argparse

import onnx
import torch

from model import LiteSRNet

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', default='checkpoints/best_model.pth')
ap.add_argument('--out', default='sr_model.onnx')
args = ap.parse_args()

model = LiteSRNet(ch=56, num_blocks=9)
ckpt = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
for name, p in model.named_parameters():  # apply EMA weights if present
    if name in ckpt.get('ema_shadow', {}):
        p.data = ckpt['ema_shadow'][name].cpu()
model.eval()

dummy = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy, args.out, input_names=['input'], output_names=['output'],
                  opset_version=11, do_constant_folding=True, dynamo=False)

# merge any external data into a single file and verify
m = onnx.load(args.out)
onnx.save_model(m, args.out, save_as_external_data=False)
onnx.checker.check_model(onnx.load(args.out))
print('exported', args.out, 'opset', onnx.load(args.out).opset_import[0].version)
