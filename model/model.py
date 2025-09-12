import torch.nn as nn
import torchvision.models as tv

NUM_CLASSES = 7

def _resnet50(pretrained: bool = True):
    try:
        return tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    except Exception:
        return tv.resnet50(pretrained=pretrained)

class Net(nn.Module):
    def __init__(self, pretrained: bool = True, drop_p: float = 0.2):
        super().__init__()
        m = _resnet50(pretrained=pretrained)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(in_features, NUM_CLASSES)
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits
