# npu_inference.py
# DATA 255 - Lab 3: NPU inference + PSNR + latency evaluation
# Model: LiteSRNet
# Runs sr_model.mxq on the NPU and measures PSNR + latency against the CPU (ONNX) baseline

import math
import os
import statistics
import time

import maccel
import numpy as np
import onnxruntime as ort
from PIL import Image

# =============================================================================
# PATHS
# =============================================================================
MXQ_PATH = 'sr_model.mxq'
ONNX_PATH = 'sr_model.onnx'
L3_LR_valid_DIR = 'LR_val'
L3_HR_valid_DIR = 'HR_val'


# =============================================================================
# Helper: load image [0,1] normalized NCHW format
# =============================================================================
def load_image(path):
    img = Image.open(path).convert('RGB').resize((256, 256), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr.transpose(2, 0, 1), 0).astype(np.float32)
    return arr


# =============================================================================
# Helper: compute PSNR
# =============================================================================
def compute_psnr(pred, target):
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse)


def gather_images(folder):
    exts = ('.png', '.jpg', '.jpeg')
    paths = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)


# =============================================================================
# Step 1: Load validation pairs
# =============================================================================
print('=' * 60)
print('DATA 255 Lab 3 - NPU Inference, PSNR & Latency')
print('Model: LiteSRNet (ch=56, blocks=9)')
print('=' * 60)

lr_files = gather_images(L3_LR_valid_DIR)
hr_files = gather_images(L3_HR_valid_DIR)

assert len(lr_files) == len(hr_files), \
    f'Mismatch: {len(lr_files)} vs {len(hr_files)}'

print(f'Validation pairs: {len(lr_files)}')
lr_images = [load_image(f) for f in lr_files]
hr_images = [load_image(f) for f in hr_files]
print(f'Loaded {len(lr_images)} pairs')
print()

# =============================================================================
# Step 2: CPU baseline (ONNX)
# =============================================================================
print('=' * 60)
print('Step 2: CPU Baseline (ONNX)')
print('=' * 60)

cpu_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = cpu_session.get_inputs()[0].name
output_name = cpu_session.get_outputs()[0].name

print(f'Input : {input_name}')
print(f'Output : {output_name}')

for _ in range(3):  # warm-up
    cpu_session.run([output_name], {input_name: lr_images[0]})

cpu_times = []
cpu_psnrs = []
for lr_img, hr_img in zip(lr_images, hr_images):
    start = time.perf_counter()
    output = cpu_session.run([output_name], {input_name: lr_img})[0]
    cpu_times.append((time.perf_counter() - start) * 1000)
    cpu_psnrs.append(compute_psnr(output, hr_img))

cpu_mean_time = statistics.mean(cpu_times)
cpu_mean_psnr = statistics.mean(cpu_psnrs)
print(f'CPU time : {cpu_mean_time:.2f} ms/image')
print(f'CPU PSNR : {cpu_mean_psnr:.3f} dB')
print()

# =============================================================================
# Step 3: NPU inference
# =============================================================================
print('=' * 60)
print('Step 3: NPU Inference (MXQ)')
print('=' * 60)

npu_device = maccel.Accelerator(0)
npu_model = maccel.Model(MXQ_PATH)
npu_model.launch(npu_device)

for _ in range(3):  # warm-up
    npu_model.infer([lr_images[0]])

npu_times = []
npu_psnrs = []
for lr_img, hr_img in zip(lr_images, hr_images):
    start = time.perf_counter()
    output = npu_model.infer([lr_img])[0]
    npu_times.append((time.perf_counter() - start) * 1000)
    npu_psnrs.append(compute_psnr(output, hr_img))

npu_mean_time = statistics.mean(npu_times)
npu_mean_psnr = statistics.mean(npu_psnrs)
npu_model.dispose()

print(f'NPU time : {npu_mean_time:.2f} ms/image')
print(f'NPU PSNR : {npu_mean_psnr:.3f} dB')
print()

# =============================================================================
# Step 4: Final results + course score estimate
# =============================================================================
print('=' * 60)
print('FINAL RESULTS')
print('=' * 60)
print('Model : LiteSRNet (ch=56, blocks=9)')
print(f'Val images : {len(lr_images)}')
print()
print(f'CPU PSNR : {cpu_mean_psnr:.3f} dB')
print(f'NPU PSNR : {npu_mean_psnr:.3f} dB')
print(f'PSNR drop : {cpu_mean_psnr - npu_mean_psnr:.3f} dB')
print()
print(f'CPU speed : {cpu_mean_time:.2f} ms/image')
print(f'NPU speed : {npu_mean_time:.2f} ms/image')
print(f'Speedup : {cpu_mean_time / npu_mean_time:.1f}x')
print()

if npu_mean_psnr > 24.3:
    if npu_mean_time < 6:
        score = 18
    elif npu_mean_time < 8:
        score = 17
    elif npu_mean_time < 10:
        score = 16
    elif npu_mean_time < 12:
        score = 15
    elif npu_mean_time < 14:
        score = 14
    elif npu_mean_time < 16:
        score = 13
    elif npu_mean_time < 18:
        score = 12
    elif npu_mean_time < 20:
        score = 11
    else:
        score = 10
elif npu_mean_psnr >= 23.3:
    score = 8
else:
    score = 0

print(f'PSNR > 24.3 : {"PASSED" if npu_mean_psnr > 24.3 else "NOT PASSED"}')
print(f'Latency : {npu_mean_time:.2f} ms/image')
print(f'Est. score : {score}/18 points')
print('=' * 60)
