import torch
from thop import profile

from src.models.standard_cnn import StandardCNN
from src.models.lightweight_cnn import LightweightCNN


def analyze_model(model, model_name):
    model.eval()

    # MNIST input: batch size 1, grayscale, 28x28
    input_tensor = torch.randn(1, 1, 28, 28)

    macs, params = profile(
        model,
        inputs=(input_tensor,),
        verbose=False
    )

    flops = macs * 2

    print("\n" + "=" * 50)
    print(f"{model_name}")
    print("=" * 50)

    print(f"Parameters : {params:,.0f}")
    print(f"MACs       : {macs:,.0f}")
    print(f"FLOPs      : {flops:,.0f}")

    print(f"MACs       : {macs / 1e6:.3f} M")
    print(f"FLOPs      : {flops / 1e6:.3f} M")


def main():
    standard_model = StandardCNN()
    lightweight_model = LightweightCNN()

    analyze_model(
        standard_model,
        "Standard CNN"
    )

    analyze_model(
        lightweight_model,
        "Lightweight CNN"
    )


if __name__ == "__main__":
    main()