import types
from typing import Any, cast, Optional, Union

import torch
import torch.nn as nn

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface

__all__ = [
    "VGG",
    "custom_vgg16_bn",
    "custom_vgg19_bn",
    "get_vgg_models"
]


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def get_classifier(self) -> nn.Sequential:
        return self.classifier

    def forward(self, x: torch.Tensor, return_feat=None) -> torch.Tensor:
        x = self.features(x) 

        feat_map = x 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = x
        x = self.classifier(x)

        if return_feat == 'feat_map':
            return x, feat_map
        elif return_feat == 'feat':
            return x, feat
        else:
            return x

def make_layers(cfg: list[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: dict[str, list[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model


_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
    "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
}



@register_model()
def custom_vgg16_bn(*, weights: None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-16-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG16_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG16_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG16_BN_Weights
        :members:
    """
    weights = None

    return _vgg("D", True, weights, progress, **kwargs)


@register_model()
def custom_vgg19_bn(*, weights: None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-19_BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG19_BN_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG19_BN_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG19_BN_Weights
        :members:
    """
    weights = None

    return _vgg("E", True, weights, progress, **kwargs)


def vgg_feature_map_hooks(model):

    def add_hooks(self, layers_to_hook):

        def _activation_hook(name, module, input, output):
            self.hooked_activations[name] = output
            self.hooked_activations[name].requires_grad_()
            self.hooked_activations[name].retain_grad()

        for name, module in self.named_modules():
            if name in layers_to_hook:
                # module.register_forward_hook(_activation_hook)
                module.register_forward_hook(lambda module, input, output, name=name: _activation_hook(name, module, input, output))

        return
    
    model.hooked_activations = {}
    model.add_hooks = types.MethodType(add_hooks, model)

    return model

def get_vgg_models(model_name: str, num_classes: int, num_channels: int) -> nn.Module:


    if model_name == 'vgg16':
        model = custom_vgg16_bn(weights=None, num_classes=num_classes)
    elif model_name == 'vgg19':
        model = custom_vgg19_bn(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not recognized. Available models: vgg16, vgg19.")
    

    model = vgg_feature_map_hooks(model)

    
    return model
