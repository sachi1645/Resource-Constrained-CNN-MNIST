import torch
from thop import profile

from src.models.emnist_standard_cnn import EMNISTStandardCNN
from src.models.emnist_lightweight_cnn import EMNISTLightweightCNN


def measure_complexity(model, model_name, device):
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(
        1, 1, 28, 28,
        device=device
    )

    macs, params = profile(
        model,
        inputs=(dummy_input,),
        verbose=False
    )

    flops = macs * 2

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Parameters : {params:,.0f}")
    print(f"MACs       : {macs:,.0f}")
    print(f"FLOPs      : {flops:,.0f}")

    print(f"MACs       : {macs / 1e6:.3f} M")
    print(f"FLOPs      : {flops / 1e6:.3f} M")


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("EMNIST COMPUTATIONAL COMPLEXITY")
    print("=" * 60)

    print(f"Device: {device}")

    standard_model = EMNISTStandardCNN(
        num_classes=47
    )

    lightweight_model = EMNISTLightweightCNN(
        num_classes=47
    )

    measure_complexity(
        standard_model,
        "EMNIST Standard CNN",
        device
    )

    measure_complexity(
        lightweight_model,
        "EMNIST Lightweight CNN",
        device
    )


if __name__ == "__main__":
    main()