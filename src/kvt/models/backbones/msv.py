import torch.nn as nn

try:
    import microsoftvision
except ImportError:
    microsoftvision = None


class ResNet(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False, in_features=2048):
        super(ResNet, self).__init__()
        self.backbone = microsoftvision.models.resnet50(pretrained=pretrained)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


def resnet50_msv(num_classes=1000, pretrained=False, **kwargs):
    model = ResNet(num_classes, pretrained, 2048)
    return model
