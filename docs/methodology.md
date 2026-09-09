# Research Methodology

## 1. Research Objective

The objective of this project is to investigate whether a compact convolutional neural network can achieve high handwritten-digit classification accuracy while significantly reducing model size and computational complexity.

Two CNN architectures are developed and compared:

- Standard CNN
- Lightweight CNN using depthwise separable convolutions

The lightweight model is subject to a strict constraint:

> Fewer than 100,000 trainable parameters.

The models are evaluated using accuracy, model size, parameter count, MACs, FLOPs, and inference latency.

---

## 2. Dataset

The project uses the MNIST handwritten digit dataset.

MNIST contains:

- 60,000 training images
- 10,000 test images
- 10 digit classes
- Image resolution of 28 × 28 pixels
- Single grayscale channel

The ten classes are:

```text
0 1 2 3 4 5 6 7 8 9