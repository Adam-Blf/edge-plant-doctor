"""Student and teacher definitions (torchvision, ImageNet weights)."""

import torchvision as tv
from torch import nn

from .config import NUM_CLASSES


def student(pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small with a 38-way head. Takes already normalised input."""
    net = tv.models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, NUM_CLASSES)
    return net


def teacher(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with a 38-way head (about 20x the compute of the student)."""
    net = tv.models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, NUM_CLASSES)
    return net


BUILDERS = {"student": student, "teacher": teacher}
