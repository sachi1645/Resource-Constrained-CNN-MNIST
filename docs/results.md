# Final Results

## 1. Overview

This project compares a conventional convolutional neural network with a lightweight CNN designed for resource-constrained deployment.

The lightweight model uses depthwise separable convolutions and is constrained to fewer than 100,000 trainable parameters.

The final comparison evaluates:

- Classification accuracy
- Test loss
- Trainable parameters
- Model size
- MACs
- FLOPs
- CPU inference latency
- GPU inference latency

---

## 2. Final Model Comparison

| Metric | Standard CNN | Lightweight CNN |
|---|---:|---:|
| Trainable Parameters | 896,906 | **3,989** |
| Model Size | 3.4214 MB | **0.0165 MB** |
| Test Accuracy | **99.05%** | 97.23% |
| Test Loss | **0.0281** | 0.0909 |
| MACs | 8.255 M | **0.376 M** |
| FLOPs | 16.510 M | **0.752 M** |
| CPU Latency | **0.613 ms** | 1.000 ms |
| GPU Latency | **0.815 ms** | 1.223 ms |

---

## 3. Parameter Efficiency

The standard CNN contains:

```text
896,906 parameters