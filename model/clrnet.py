import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_implict import *
from .cct import *
from .resnet18_sp import *
from .resnet18_pool import *
from .resnet_nopool import *

__all__ = ['CLRBackbone', 'CLRLinearClassifier']

model_dict = {
    'resnet18_no_pool': [resnet18_no_pool(True), 512],
    'cct_7_7x2_224': [cct_7_7x2_224(num_classes=512), 512],
    'resnet18_sp': [resnet18_sp(True), 512],
    'resnet18_learnpool': [resnet18_learnpool(True, 512), 512],
    'resnet18': [resnet18(True), 512],
    'resnet18_LN': [resnet18_LN(True), 512],
    'resnet18_GELU': [resnet18_GELU(True), 512],
    'implicit_resnet18': [implicit_resnet18(pretrained=True), 512],
    'implicit_resnet18_last_mul_add': [implicit_resnet18_last_mul_add(pretrained=True), 512],
    'implicit_resnet18_mul_last_four': [implicit_resnet18_mul_last_four_layer(True), 512],
    'implicit_resnet18_add_last_four': [implicit_resnet18_add_last_four_layer(True), 512],
    'implicit_resnet18_add_all': [implicit_resnet18_add_all_layer(True), 512],
    'implicit_resnet18_mul_add_every': [implicit_resnet18_mul_add_every(True), 512],
    'implicit_resnet18_m_a_m_a_2_3_4_5': [implicit_resnet18_m_a_m_a_2_3_4_5(True), 512],
    'implicit_resnet18_m_m_a_a_2_3_4_5': [implicit_resnet18_m_m_a_a_2_3_4_5(True), 512],
    'implicit_resnet18_mul_add_5_output': [implicit_resnet18_mul_add_5_output(True), 512],
    'implicit_resnet18_mul_add_4_5_output': [implicit_resnet18_mul_add_4_5_output(True), 512],
    'implicit_resnet18_mul_add_3_4_5_output': [implicit_resnet18_mul_add_3_4_5_output(True), 512],
    'implicit_resnet18_a_a_m_m_m_2_3_4_5_pool': [implicit_resnet18_a_a_m_m_m_2_3_4_5_pool(True), 512],
    'implicit_resnet18_a_a_m_m_a_2_3_4_5_pool': [implicit_resnet18_a_a_m_m_a_2_3_4_5_pool(True), 512],
    'implicit_resnet18_m_a_m_a_m_2_3_4_5_pool': [implicit_resnet18_m_a_m_a_m_2_3_4_5_pool(True), 512],
    'implicit_resnet18_m_5': [implicit_resnet18_m_5(True), 512],
    'implicit_resnet18_m_a_2': [implicit_resnet18_m_5(True), 512],
    'implicit_resnet18_m_a_3': [implicit_resnet18_m_a_3(True), 512],
    'implicit_resnet18_m_a_4': [implicit_resnet18_m_a_4(True), 512],
    'implicit_resnet18_m_a_5': [implicit_resnet18_m_a_5(True), 512],
    'implicit_resnet18_m_m_a_a_2_3_4_5_GELU': [implicit_resnet18_m_m_a_a_2_3_4_5_GELU(True), 512],
}


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
        if model_fun is None:
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
