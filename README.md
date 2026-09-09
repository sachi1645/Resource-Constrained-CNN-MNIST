# Resource-Constrained CNN for MNIST Character Classification

A PyTorch research project comparing a standard CNN with a lightweight CNN based on depthwise separable convolutions.

## Main objectives

- Train a standard CNN baseline.
- Design a lightweight CNN for resource-constrained environments.
- Keep the lightweight model below 100,000 trainable parameters.
- Compare accuracy, parameter count, model size, inference time, and memory usage.
- Deploy the lightweight model in an interactive handwritten-digit drawing application.

## Project pipeline

MNIST -> preprocessing -> CNN -> evaluation -> lightweight model -> drawing application

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python -m src.training.train_standard
python -m src.training.train_lightweight
```

## Evaluate

```powershell
python -m src.evaluation.evaluate --model lightweight
python -m src.evaluation.evaluate --model standard
```

## Benchmark

```powershell
python -m src.evaluation.benchmark
```

## Run the drawing application

After training the lightweight model:

```powershell
python app/app.py
```

Draw a digit with the mouse and press **Predict**.

## Research constraint

The lightweight model contains fewer than 100,000 trainable parameters. The training script checks this constraint before training.
