from .resnet import ResNet50Classifier
from .efficientnet import EfficientNetB0Classifier
from .global_stats import GlobalStatsDecoder
from .spectral import SpectralDecoder
from .multiscale import MultiScalePyramidDecoder
from .dual_branch import DualBranchDecoder


_BACKBONELESS = {"global_stats", "spectral"}


def get_model(architecture, num_outputs=8, pretrained=True):
    architecture = architecture.lower()
    if architecture == 'resnet50':
        return ResNet50Classifier(num_outputs=num_outputs, pretrained=pretrained)
    if architecture == 'efficientnet_b0':
        return EfficientNetB0Classifier(num_outputs=num_outputs, pretrained=pretrained)
    if architecture == 'global_stats':
        return GlobalStatsDecoder(num_outputs=num_outputs)
    if architecture == 'spectral':
        return SpectralDecoder(num_outputs=num_outputs)
    if architecture in ('multiscale', 'multiscale_pyramid'):
        return MultiScalePyramidDecoder(num_outputs=num_outputs, pretrained=pretrained)
    if architecture == 'dual_branch':
        return DualBranchDecoder(num_outputs=num_outputs, pretrained=pretrained)
    if architecture in ('dual_branch_r34', 'dual_branch_resnet34'):
        return DualBranchDecoder(num_outputs=num_outputs, pretrained=pretrained,
                                 spatial_backbone="resnet34")
    if architecture in ('dual_branch_r50', 'dual_branch_resnet50'):
        return DualBranchDecoder(num_outputs=num_outputs, pretrained=pretrained,
                                 spatial_backbone="resnet50")
    raise ValueError(f"Unknown architecture: {architecture}")


__all__ = [
    "ResNet50Classifier",
    "EfficientNetB0Classifier",
    "GlobalStatsDecoder",
    "SpectralDecoder",
    "MultiScalePyramidDecoder",
    "DualBranchDecoder",
    "get_model",
]
