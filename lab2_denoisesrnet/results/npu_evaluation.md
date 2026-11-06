# Lab 2 NPU evaluation (100 validation images, 256x256)

Transcribed from the terminal screenshot saved with the lab submission (output of `npu_inference.py`).

| Metric | CPU (ONNX Runtime) | NPU (MXQ) |
|---|---|---|
| Mean PSNR | 22.423 dB | 22.400 dB |
| Latency | 1578.60 ms/image | 86.07 ms/image (11.6 img/s) |
| Speedup | | 18.3x |

- Model: DenoiseSRNet, 2,751,747 parameters
- PSNR drop from quantization: 0.023 dB
- Course scoring band printed by the script: 14 / 19 (PSNR above 22 dB)
