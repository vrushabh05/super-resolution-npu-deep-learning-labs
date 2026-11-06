# Lab 3 NPU evaluation (110 validation images, 256x256)

Output of `npu_inference.py` on the Mobilint NPU, transcribed from the terminal screenshot saved with the lab submission.

| Metric | CPU (ONNX Runtime) | NPU (MXQ) |
|---|---|---|
| Mean PSNR | 24.330 dB | 24.283 dB |
| Latency | 35.76 ms/image | 8.57 ms/image |
| Speedup | | 4.2x |

- PSNR drop from quantization: 0.048 dB
- Course target "PSNR > 24.3 dB": **not passed on the NPU** (24.283 dB), passed on CPU (24.330 dB)
- Course estimate printed by the script: 8 / 18 points (the 23.3 to 24.3 dB band)
