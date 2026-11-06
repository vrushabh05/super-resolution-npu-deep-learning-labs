# onnx_to_mxq.py
# DATA 255 - Lab 2: ONNX to MXQ conversion
# Converts sr_model.onnx -> sr_model.mxq for NPU execution.
# Needs the Mobilint qubee toolchain.

import os
import shutil

import numpy as np
from PIL import Image
from qubee import mxq_compile
from qubee.calibration import make_calib_man

# Paths
ONNX_PATH = 'sr_model.onnx'
MXQ_PATH = 'sr_model.mxq'
CALIB_IMAGE_DIR = 'calibration_images'
CALIB_DATA_DIR = 'calibration_data_sr'


def preprocess_sr(image_path):
    """Matches training normalization [0, 1], input size 256x256x3."""
    img = Image.open(image_path).convert('RGB').resize((256, 256), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# Step 1: build calibration data from images
print('=' * 60)
print('Step 1: Building calibration data from images...')
print('=' * 60)

if os.path.exists(CALIB_DATA_DIR):
    shutil.rmtree(CALIB_DATA_DIR)
    print(f'Removed old: {CALIB_DATA_DIR}')

calib_images = [
    f for f in os.listdir(CALIB_IMAGE_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
print(f'Found {len(calib_images)} calibration images in {CALIB_IMAGE_DIR}/')

make_calib_man(
    pre_ftn=preprocess_sr,
    data_dir=CALIB_IMAGE_DIR,
    save_dir='.',
    save_name=CALIB_DATA_DIR,
    max_size=len(calib_images),
)
print(f'Calibration data built: {CALIB_DATA_DIR}')

# Step 2: compile ONNX to MXQ
print()
print('=' * 60)
print('Step 2: Compiling ONNX to MXQ...')
print('=' * 60)
print(f'Input : {ONNX_PATH}')
print(f'Output : {MXQ_PATH}')
print('Model : DenoiseSRNet (2,751,747 params)')
print('Input size: [1, 3, 256, 256]')

mxq_compile(
    model=ONNX_PATH,
    calib_data_path=CALIB_DATA_DIR,
    quantization_mode=2,        # maxPercentile
    is_quant_ch=True,
    percentile=0.999,
    topk_ratio=0.01,
    quantization_output=0,      # layer quantization
    save_path=MXQ_PATH,
    backend='onnx',
)

# Step 3: verify
print()
print('=' * 60)
print('Step 3: Verifying output...')
print('=' * 60)

if os.path.exists(MXQ_PATH):
    size_mb = os.path.getsize(MXQ_PATH) / (1024 * 1024)
    print(f'MXQ file created : {MXQ_PATH}')
    print(f'MXQ file size : {size_mb:.1f} MB')
    print('Compilation successful')
    print()
    print('Next: run npu_inference.py to evaluate PSNR on the NPU')
else:
    print('ERROR: MXQ file not created')
