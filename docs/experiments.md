# Experimental Results

## 1. Experimental Objective

The experiments evaluate whether a highly compact CNN can maintain competitive MNIST classification accuracy while significantly reducing:

- Trainable parameters
- Model size
- MACs
- FLOPs

The experiments also measure real inference latency on CPU and GPU.

Two models are compared:

1. Standard CNN
2. Lightweight CNN

The lightweight CNN must satisfy:

> Fewer than 100,000 trainable parameters.

---

## 2. Experimental Environment

The experiments were performed using:

```text
Operating System : Windows 11
GPU              : NVIDIA GeForce RTX 2050 4GB
CUDA             : 12.6
PyTorch          : 2.10.0+cu126
Python           : Python 3