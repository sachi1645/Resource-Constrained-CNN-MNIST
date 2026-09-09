# Project Architecture

## 1. Project Overview

This project develops and evaluates convolutional neural networks (CNNs) for handwritten digit classification using the MNIST dataset.

The main research objective is to compare:

1. A standard CNN architecture.
2. A lightweight CNN based on depthwise separable convolutions.

The lightweight model is designed for resource-constrained and edge-computing environments.

A strict research constraint is applied:

> The lightweight CNN must contain fewer than 100,000 trainable parameters.

The final system also includes an interactive desktop application that allows a user to draw a handwritten digit and receive a prediction from the trained lightweight CNN.

---

## 2. High-Level Architecture

The overall system can be represented as:

```text
                    MNIST Dataset
                         |
                         v
                +-------------------+
                | Data Preprocessing|
                +-------------------+
                         |
                         v
              +---------------------+
              | Train / Validation  |
              |       Split         |
              +---------------------+
                    /          \
                   /            \
                  v              v
        +---------------+  +-------------------+
        | Standard CNN  |  | Lightweight CNN   |
        +---------------+  +-------------------+
                |                  |
                v                  v
        +---------------+  +-------------------+
        |   Evaluation  |  |   Evaluation      |
        +---------------+  +-------------------+
                |                  |
                +--------+---------+
                         |
                         v
                +-------------------+
                | Model Comparison  |
                +-------------------+
                         |
                         v
                Accuracy / Size /
                Parameters / MACs /
                FLOPs / Latency