"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
import torch.nn as nn

# BUG FIX: changed from "Identity" to "ReLU"
# Identity does nothing — network could not learn non-linear patterns
activation_str = "ReLU"


class VGGBlock(nn.Module):
    """Modular VGG block with configurable number of conv layers and channels.
    C configuration from Simonyan & Zisserman VGG paper.
    """
    def __init__(self, in_channels, out_channels, num_convs, padding=1):
        super().__init__()
        layers = []
        current_in_channels = in_channels
        for i in range(num_convs):
            is_config_c_tail = (num_convs == 3 and i == 2)
            kernel_size = 1 if is_config_c_tail else 3
            # BUG FIX: padding was always 1 even for 1x1 kernels
            # 1x1 conv needs padding=0 otherwise spatial dimensions grow incorrectly
            pad = 0 if is_config_c_tail else padding
            layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=kernel_size, padding=pad))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            # BUG FIX: current_in_channels was never updated inside the loop
            # next conv layer needs to know output channels of previous conv
            current_in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """ResBlock with 3x3 convolutions (He et al., 2016)."""
    def __init__(self, in_channels, out_channels, activation, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.activation(out)
        return out


class AlexNet(nn.Module):
    """AlexNet (Krizhevsky et al., 2012) adapted for smaller inputs."""
    # BUG FIX: original only accepted **kwargs and ignored in_channels and num_classes
    # hardcoded 3 channels and 11 classes crashed on grayscale datasets
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)

        self.features = nn.Sequential(
            # BUG FIX: was hardcoded as nn.Conv2d(3, 48, ...) ignoring in_channels
            nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            # BUG FIX: was nn.Linear(2048, 1024) — wrong size for 64x64 input
            # 192 channels x 4 x 4 spatial = 3072 features after flattening
            nn.Linear(3072, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            # BUG FIX: was hardcoded nn.Linear(1024, 11) ignoring num_classes
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGG16(nn.Module):
    """VGG16 in C configuration (Simonyan & Zisserman, 2014) adapted for smaller inputs."""
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)

        self.features = nn.Sequential(
            VGGBlock(in_channels, 64, num_convs=2),
            VGGBlock(64, 128, num_convs=2),
            VGGBlock(128, 256, num_convs=3),
            VGGBlock(256, 512, num_convs=3),
            VGGBlock(512, 512, num_convs=3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet18(nn.Module):
    """ResNet18 (He et al., 2016) adapted for smaller inputs.
    activation — flexible activation function for experimentation.
    """
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        activation = getattr(nn, activation_str)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = activation(inplace=True)
        print("Using activation function:", self.activation)

        self.stage1 = nn.Sequential(
            ResBlock(64, 64, activation(inplace=True), stride=1),
            ResBlock(64, 64, activation(inplace=True), stride=1)
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, 128, activation(inplace=True), stride=2),
            ResBlock(128, 128, activation(inplace=True), stride=1)
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, activation(inplace=True), stride=2),
            ResBlock(256, 256, activation(inplace=True), stride=1)
        )
        self.stage4 = nn.Sequential(
            ResBlock(256, 512, activation(inplace=True), stride=2),
            ResBlock(512, 512, activation(inplace=True), stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # BUG FIX: was missing return — model was outputting None silently
        return self.classifier(out)


class PlantNet(nn.Module):
    """PlantNet — lightweight green initiative model.
    Designed for energy-efficient deployment on diagnostic devices.
    4 conv blocks with progressive channel expansion: 16 -> 32 -> 64 -> 64
    """
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.3)

        self.features = nn.Sequential(
            # Block 1 — detects basic features like edges and corners
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2 — detects medium features like textures and curves
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3 — detects complex features like shapes and patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4 — high level feature summarization
            # AdaptiveAvgPool squeezes spatial dims to 2x2 regardless of input size
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            # 64 channels x 2 x 2 spatial = 256 input features
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)