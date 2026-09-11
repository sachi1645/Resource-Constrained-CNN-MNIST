import time

import torch

from src.models.emnist_standard_cnn import EMNISTStandardCNN
from src.models.emnist_lightweight_cnn import EMNISTLightweightCNN


def benchmark_model(model, model_name, device, runs=500, warmup=20):
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(
        1, 1, 28, 28,
        device=device
    )

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    latency_ms = (
        elapsed / runs
    ) * 1000

    print(
        f"{model_name:<25} "
        f"{latency_ms:.4f} ms/image"
    )

    return latency_ms


def run_benchmark(device):

    print("\n" + "=" * 60)
    print(f"{device.type.upper()} BENCHMARK")
    print("=" * 60)

    standard_model = EMNISTStandardCNN(
        num_classes=47
    )

    lightweight_model = EMNISTLightweightCNN(
        num_classes=47
    )

    standard_latency = benchmark_model(
        standard_model,
        "Standard CNN",
        device
    )

    lightweight_latency = benchmark_model(
        lightweight_model,
        "Lightweight CNN",
        device
    )

    print("\nLatency ratio:")

    print(
        f"Lightweight / Standard: "
        f"{lightweight_latency / standard_latency:.2f}x"
    )


def main():

    print("=" * 60)
    print("EMNIST INFERENCE BENCHMARK")
    print("=" * 60)

    # CPU
    run_benchmark(
        torch.device("cpu")
    )

    # GPU
    if torch.cuda.is_available():
        run_benchmark(
            torch.device("cuda")
        )


if __name__ == "__main__":
    main()