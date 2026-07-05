import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class MobileNetBackbone(nn.Module):
    def __init__(self, device='cpu', output_dim=128, pretrained = False, freeze=False, unfreeze_last_n=1):
        super(MobileNetBackbone, self).__init__()
        self.device = device
        output_dim = int(output_dim)
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.flatten = nn.Flatten()

        # Freeze all layers first
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.avgpool.parameters():
                param.requires_grad = False

            # Unfreeze last `n` blocks
            if unfreeze_last_n > 0:
                blocks = list(self.features.children())
                for block in blocks[-unfreeze_last_n:]:
                    for param in block.parameters():
                        param.requires_grad = True

        # Final projection to output_dim
        # Final projection to output_dim (no activation here -- produce raw features/logits)
        self.class_head = nn.Sequential(mobilenet.classifier[0], mobilenet.classifier[1], mobilenet.classifier[2])
        self.dropout = nn.Dropout(.2)
        self.fc = nn.Linear(mobilenet.classifier[3].in_features, output_dim)

    def forward(self, x):
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.class_head(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def get_mobilenet_models(model_name: str, num_classes: int, num_channels: int) -> nn.Module:


    model = MobileNetBackbone(pretrained=None, output_dim=num_classes)


    
    return model
