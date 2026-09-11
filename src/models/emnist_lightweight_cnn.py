import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EMNISTLightweightCNN(nn.Module):
    """
    Lightweight CNN for 47-class EMNIST classification.

    Designed to remain below 100,000 trainable parameters.
    """

    def __init__(self, num_classes=47):
        super().__init__()

        self.features = nn.Sequential(
            DepthwiseSeparableConv(1, 16),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(16, 32),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(32, 64),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x