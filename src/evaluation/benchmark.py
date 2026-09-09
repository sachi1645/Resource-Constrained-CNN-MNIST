import json
import time
from pathlib import Path

import psutil
import torch

from src.models.lightweight_cnn import LightweightCNN
from src.models.standard_cnn import StandardCNN
from src.models.model_utils import (
    load_checkpoint,
    count_parameters,
    model_size_mb,
)


ROOT = Path(__file__).resolve().parents[2]


def benchmark_model(model, device, runs=500):
    """
    Measure average single-image inference latency.
    """

    model.eval()
    model.to(device)

    sample = torch.randn(1, 1, 28, 28, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(20):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(runs):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return (elapsed / runs) * 1000


def run_benchmark(device):
    """
    Benchmark both models on the selected device.
    """

    print("\n" + "=" * 60)
    print(f"Benchmarking on: {device}")
    print("=" * 60)

    models = [
        (
            "standard",
            StandardCNN(),
            ROOT / "models" / "standard_cnn" / "best_model.pth",
        ),
        (
            "lightweight",
            LightweightCNN(),
            ROOT / "models" / "lightweight_cnn" / "best_model.pth",
        ),
    ]

    results = []

    for name, model, path in models:

        if not path.exists():
            print(f"Skipping {name}: {path} does not exist.")
            continue

        load_checkpoint(model, path, device)

        latency = benchmark_model(model, device)

        result = {
            "device": str(device),
            "model": name,
            "trainable_parameters": count_parameters(model),
            "model_size_mb": round(model_size_mb(model), 6),
            "average_inference_ms": round(latency, 6),
            "system_memory_percent": psutil.virtual_memory().percent,
        }

        results.append(result)

        print(f"\nModel: {name}")
        print(f"Parameters: {result['trainable_parameters']:,}")
        print(f"Model size: {result['model_size_mb']} MB")
        print(f"Average inference: {result['average_inference_ms']} ms/image")

    return results


def main():

    # Always benchmark CPU
    cpu_results = run_benchmark(torch.device("cpu"))

    # Benchmark GPU if CUDA is available
    gpu_results = []

    if torch.cuda.is_available():
        gpu_results = run_benchmark(torch.device("cuda"))
    else:
        print("\nCUDA is not available. GPU benchmark skipped.")

    all_results = cpu_results + gpu_results

    output = ROOT / "results" / "benchmark_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 60)

    print(json.dumps(all_results, indent=2))

    print(f"\nSaved benchmark results to:")
    print(output)


if __name__ == "__main__":
    main()