import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from .resnet_implict import implicit_resnet50, implicit_resnext50_32x4d
from .resnet_big import resnet50
from .coatnet_slr import coatnet_0, coatnet_1

__all__ = ['CLRBackbone', 'CLRLinearClassifier']

model_dict = {
    'coatnet_0': [coatnet_0(), 768],
    'coatnet_1': [coatnet_1(), 768],
    'resnet50': [resnet50(), 2048],
    'implicit_resnet50': [implicit_resnet50(pretrained=True), 2048],
    'implicit_resnext50': [implicit_resnext50_32x4d(pretrained=True), 2048],

}


# TODO: Normalisze position have an effect on result?

class Norm(nn.Module):
    def __init__(self, p: float = 2.0):
        super(Norm, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(input=x, p=self.p, dim=1)


class CLRHead(nn.Module):
    def __init__(self,
                 head: str = 'mlp',
                 dim_in: int = 2048,
                 feat_dim: int = 128):
        super(CLRHead, self).__init__()
        self.head = nn.ModuleList([])
        self.head.append(nn.Linear(dim_in, dim_in))

        if head == 'mlp':
            self.head.append(nn.ReLU(inplace=True))
        elif head == 'mlp_bn':
            self.head.append(nn.Sequential(
                nn.BatchNorm1d(dim_in),
                nn.ReLU(inplace=True)))
        elif head == '2mlp_bn':
            self.head.append(nn.Sequential(
                nn.BatchNorm1d(dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_in),
                nn.BatchNorm1d(dim_in),
                nn.ReLU(inplace=True),
            ))
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        self.head.append(nn.Linear(dim_in, feat_dim))
        self.head.append(Norm(p=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.head:
            x = layer(x)
        return x


class CLRBackbone(nn.Module):
    def __init__(self, name: str = 'resnet50',
                 head: str = 'mlp',
                 feat_dim: int = 128):
        super(CLRBackbone, self).__init__()
        model_fun, dim_in = model_dict[name]
        if model_fun == None:
            raise NotImplementedError(f'Backbone not supperted:{model_fun}')
        self.encoder = model_fun
        self.head = CLRHead(head=head, dim_in=dim_in, feat_dim=feat_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


class CLRLinearClassifier(nn.Module):
    def __init__(self, name: str = 'resnet50',
                 num_classes: int = 10):
        super(CLRLinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class CLRClassifier(nn.Module):
    def __init__(self, name: str = 'resnet50',
                 num_classes: int = 10):
        super(CLRClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        return self.classifier(x)
