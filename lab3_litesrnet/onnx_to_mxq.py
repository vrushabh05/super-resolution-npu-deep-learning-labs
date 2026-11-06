# onnx_to_mxq.py
# DATA 255 - Lab 3: ONNX to MXQ conversion script
# Model: LiteSRNet
# Converts sr_model.onnx -> sr_model.mxq for NPU execution (run inside the Mobilint qubee environment)

import os
import shutil

import numpy as np
from PIL import Image
from qubee import mxq_compile
from qubee.calibration import make_calib_man

# =============================================================================
# PATHS
# =============================================================================
ONNX_PATH = 'sr_model.onnx'
MXQ_PATH = 'sr_model.mxq'
CALIB_IMAGE_DIR = 'calibration_images'
CALIB_DATA_DIR = 'calibration_data_sr'


# =============================================================================
# Preprocessing - matches training normalization [0,1]
# =============================================================================
def preprocess_sr(image_path):
    img = Image.open(image_path).convert('RGB').resize((256, 256), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# =============================================================================
# Step 1: Build calibration data from images
# =============================================================================
print('=' * 60)
print('Step 1: Building calibration data...')
print('=' * 60)

if os.path.exists(CALIB_DATA_DIR):
    shutil.rmtree(CALIB_DATA_DIR)

calib_images = [
    f for f in os.listdir(CALIB_IMAGE_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
print(f'Found {len(calib_images)} calibration images')

make_calib_man(
    pre_ftn=preprocess_sr,
    data_dir=CALIB_IMAGE_DIR,
    save_dir='.',
    save_name=CALIB_DATA_DIR,
    max_size=len(calib_images),
)
print(f'Calibration data built: {CALIB_DATA_DIR}')


# =============================================================================
# Step 2: Compile ONNX to MXQ
# =============================================================================
print()
print('=' * 60)
print('Step 2: Compiling ONNX to MXQ...')
print('=' * 60)
print(f'Input : {ONNX_PATH}')
print(f'Output : {MXQ_PATH}')
print('Model : LiteSRNet (ch=56, blocks=9)')

mxq_compile(
    model=ONNX_PATH,
    calib_data_path=CALIB_DATA_DIR,
    quantization_mode=2,
    is_quant_ch=True,
    percentile=0.999,
    topk_ratio=0.01,
    quantization_output=0,
    save_path=MXQ_PATH,
    backend='onnx',
)


# =============================================================================
# Step 3: Verify
# =============================================================================
print()
print('=' * 60)
print('Step 3: Verifying output...')
print('=' * 60)

if os.path.exists(MXQ_PATH):
    size_mb = os.path.getsize(MXQ_PATH) / (1024 * 1024)
    print(f'MXQ created : {MXQ_PATH}')
    print(f'MXQ size : {size_mb:.1f} MB')
    print('Compilation successful!')
    print()
    print('Next: Run npu_inference.py to evaluate PSNR + Latency on NPU')
else:
    print('ERROR: MXQ not created!')
