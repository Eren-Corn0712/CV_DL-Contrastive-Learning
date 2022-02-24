import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional, Tuple

from torch.hub import load_state_dict_from_url

__all__ = ['resnet18',
           'resnet18_GELU',
           'resnet18_LN',
           'implicit_resnet18',
           'implicit_resnet18_last_mul_add',
           'implicit_resnet18_add_all_layer',
           'implicit_resnet18_add_last_four_layer',
           'implicit_resnet18_mul_last_four_layer',
           'implicit_resnet18_mul_add_every',
           'implicit_resnet18_m_a_m_a_2_3_4_5',
           'implicit_resnet18_mul_add_5_output',
           'implicit_resnet18_mul_add_4_5_output',
           'implicit_resnet18_mul_add_3_4_5_output',
           'implicit_resnet18_m_m_a_a_2_3_4_5',
           'implicit_resnet18_a_a_m_m_2_3_4_5',
           'implicit_resnet18_a_a_m_m_a_2_3_4_5_pool',
           'implicit_resnet18_a_a_m_m_m_2_3_4_5_pool',
           'implicit_resnet18_m_a_m_a_m_2_3_4_5_pool',
           'implicit_resnet18_m_5',
           'implicit_resnet18_m_a_2',
           'implicit_resnet18_m_a_3',
           'implicit_resnet18_m_a_4',
           'implicit_resnet18_m_a_5',
           'implicit_resnet18_m_m_a_a_2_3_4_5_GELU']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ImplicitMul(nn.Module):
    def __init__(self, channel):
        super(ImplicitMul, self).__init__()
        self.channel = channel
        self.implicit = nn.parameter.Parameter(torch.Tensor(1, channel), requires_grad=True)
        nn.init.normal_(self.implicit, std=.02)

    def forward(self, x):
        return self.implicit.expand_as(x) * x


class ImplicitAdd(nn.Module):
    def __init__(self, channel):
        super(ImplicitAdd, self).__init__()
        self.channel = channel
        self.implicit = nn.parameter.Parameter(torch.Tensor(1, channel), requires_grad=True)
        nn.init.normal_(self.implicit, std=.02)

    def forward(self, x):
        return self.implicit.expand_as(x) + x


class ImplicitMul2D(nn.Module):
    def __init__(self, channel):
        super(ImplicitMul2D, self).__init__()
        self.channel = channel
        self.implicit = nn.parameter.Parameter(torch.Tensor(1, channel, 1, 1), requires_grad=True)
        nn.init.normal_(self.implicit, std=.02)

    def forward(self, x):
        return self.implicit.expand_as(x) * x


class ImplicitAdd2D(nn.Module):
    def __init__(self, channel):
        super(ImplicitAdd2D, self).__init__()
        self.channel = channel
        self.implicit = nn.parameter.Parameter(torch.Tensor(1, channel, 1, 1), requires_grad=True)
        nn.init.normal_(self.implicit, std=.02)

    def forward(self, x):
        return self.implicit.expand_as(x) + x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            act_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.act = act_layer()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            act_layer: Optional[Callable[..., nn.Module]] = None,
            forward_choice: str = None,
            conv1_k: Tuple[int, int] = (7, 7),
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU

        self._norm_layer = norm_layer
        self._act_layer = act_layer
        # For implicit knowledge
        self.forward_choice = forward_choice

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=conv1_k, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act = act_layer()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layer1_add = ImplicitAdd2D(64)
        self.layer1_mul = ImplicitMul2D(64)
        self.layer2_add = ImplicitAdd2D(128)
        self.layer2_mul = ImplicitMul2D(128)
        self.layer3_add = ImplicitAdd2D(256)
        self.layer3_mul = ImplicitMul2D(256)
        self.layer4_add = ImplicitAdd2D(512)
        self.layer4_mul = ImplicitMul2D(512)

        self.implicitadd = ImplicitAdd(channel=512)
        self.implicitmul = ImplicitMul(channel=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        act_layer = self._act_layer

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, act_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, act_layer=act_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _forward_add_per_layer(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_add(x)
        x = self.layer2(x)
        x = self.layer2_add(x)
        x = self.layer3(x)
        x = self.layer3_add(x)
        x = self.layer4(x)
        x = self.layer4_add(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _forward_mul_per_layer(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_mul(x)
        x = self.layer2(x)
        x = self.layer2_mul(x)
        x = self.layer3(x)
        x = self.layer3_mul(x)
        x = self.layer4(x)
        x = self.layer4_mul(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _forward_mul_last_two_layer(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer4_mul(x)  # 1

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)  # 2
        return x

    def _forward_mul_last_three_layer(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer3_mul(x)  # 1
        x = self.layer4(x)
        x = self.layer4_mul(x)  # 2

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)  # 3
        return x

    def _forward_mul_last_four_layer(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_mul(x)  # 1
        x = self.layer3(x)
        x = self.layer3_mul(x)  # 2
        x = self.layer4(x)
        x = self.layer4_mul(x)  # 3

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)  # 4
        return x

    def _forward_mul_all_layer(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_mul(x)  # 1
        x = self.layer2(x)
        x = self.layer2_mul(x)  # 2
        x = self.layer3(x)
        x = self.layer3_mul(x)  # 3
        x = self.layer4(x)
        x = self.layer4_mul(x)  # 4

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)  # 5
        return x

    def _forward_add_all_layer(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_add(x)  # 1
        x = self.layer2(x)
        x = self.layer2_add(x)  # 2
        x = self.layer3(x)
        x = self.layer3_add(x)  # 3
        x = self.layer4(x)
        x = self.layer4_add(x)  # 4

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitadd(x)  # 5
        return x

    def _forward_add_last_two_layer(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer4_add(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitadd(x)
        return x

    def _forward_add_last_three_layer(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer3_add(x)  # 1
        x = self.layer4(x)
        x = self.layer4_add(x)  # 2

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitadd(x)  # 3
        return x

    def _forward_add_last_four_layer(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_add(x)  # 1
        x = self.layer3(x)
        x = self.layer3_add(x)  # 2
        x = self.layer4(x)
        x = self.layer4_add(x)  # 3

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitadd(x)  # 4
        return x

    def _forward_last_mul_add(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)
        x = self.implicitadd(x)
        return x

    def _forward_mul_add_every(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_mul(x)
        x = self.layer1_add(x)

        x = self.layer2(x)
        x = self.layer2_mul(x)
        x = self.layer2_add(x)

        x = self.layer3(x)
        x = self.layer3_mul(x)
        x = self.layer3_add(x)

        x = self.layer4(x)
        x = self.layer4_mul(x)
        x = self.layer4_add(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)
        x = self.implicitadd(x)
        return x

    def _forward_m_a_m_a_2_3_4_5(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_mul(x)

        x = self.layer2(x)
        x = self.layer2_add(x)

        x = self.layer3(x)
        x = self.layer3_mul(x)

        x = self.layer4(x)
        x = self.layer4_add(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _forward_m_m_a_a_2_3_4_5(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_mul(x)

        x = self.layer2(x)
        x = self.layer2_mul(x)

        x = self.layer3(x)
        x = self.layer3_add(x)

        x = self.layer4(x)
        x = self.layer4_add(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _forward_a_a_m_m_2_3_4_5(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_add(x)

        x = self.layer2(x)
        x = self.layer2_add(x)

        x = self.layer3(x)
        x = self.layer3_mul(x)

        x = self.layer4(x)
        x = self.layer4_mul(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _forward_a_a_m_m_m_2_3_4_5_pool(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_add(x)

        x = self.layer2(x)
        x = self.layer2_add(x)

        x = self.layer3(x)
        x = self.layer3_mul(x)

        x = self.layer4(x)
        x = self.layer4_mul(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)

        return x

    def _forward_m_a_m_a_m_2_3_4_5_pool(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_mul(x)

        x = self.layer2(x)
        x = self.layer2_add(x)

        x = self.layer3(x)
        x = self.layer3_mul(x)

        x = self.layer4(x)
        x = self.layer4_add(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)

        return x

    def _forward_m_5(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer4_mul(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _forward_m_a_5(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer4_mul(x)
        x = self.layer4_add(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _forward_m_a_4(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer3_mul(x)
        x = self.layer3_add(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _forward_m_a_3(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_mul(x)
        x = self.layer2_add(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _forward_m_a_2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer1_mul(x)
        x = self.layer1_add(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def _forward_mul_add_5_output(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer4_mul(x)
        x = self.layer4_add(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)
        x = self.implicitadd(x)
        return x

    def _forward_mul_add_4_5_output(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer3_mul(x)
        x = self.layer3_add(x)
        x = self.layer4(x)
        x = self.layer4_mul(x)
        x = self.layer4_add(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)
        x = self.implicitadd(x)
        return x

    def _forward_mul_add_3_4_5_output(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_mul(x)
        x = self.layer2_add(x)
        x = self.layer3(x)
        x = self.layer3_mul(x)
        x = self.layer3_add(x)
        x = self.layer4(x)
        x = self.layer4_mul(x)
        x = self.layer4_add(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.implicitmul(x)
        x = self.implicitadd(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.forward_choice is None:
            return self._forward_impl(x)

        elif self.forward_choice == "add_per_layer":
            return self._forward_add_per_layer(x)

        elif self.forward_choice == "mul_per_layer":
            return self._forward_mul_per_layer(x)

        elif self.forward_choice == "mul_last_two_layer":
            return self._forward_mul_last_two_layer(x)

        elif self.forward_choice == "mul_last_three_layer":
            return self._forward_mul_last_three_layer(x)

        elif self.forward_choice == "add_last_two_layer":
            return self._forward_add_last_two_layer(x)

        elif self.forward_choice == "add_last_three_layer":
            return self._forward_add_last_three_layer(x)

        elif self.forward_choice == "last_mul_add":
            return self._forward_last_mul_add(x)

        elif self.forward_choice == "mul_last_four_layer":
            return self._forward_mul_last_four_layer(x)

        elif self.forward_choice == "mul_all_layer":
            return self._forward_mul_all_layer(x)

        elif self.forward_choice == "add_last_four_layer":
            return self._forward_add_last_four_layer(x)

        elif self.forward_choice == "add_all_layer":
            return self._forward_add_all_layer(x)

        elif self.forward_choice == "mul_add_every":
            return self._forward_mul_add_every(x)

        elif self.forward_choice == "m_a_m_a_2_3_4_5":
            return self._forward_m_a_m_a_2_3_4_5(x)

        elif self.forward_choice == 'm_m_a_a_2_3_4_5':
            return self._forward_m_m_a_a_2_3_4_5(x)

        elif self.forward_choice == "a_a_m_m_2_3_4_5":
            return self._forward_a_a_m_m_2_3_4_5(x)

        elif self.forward_choice == "a_a_m_m_m_2_3_4_5_pool":
            return self._forward_a_a_m_m_m_2_3_4_5_pool(x)  # 2022/2/23

        elif self.forward_choice == "a_a_m_m_a_2_3_4_5_pool":
            return self._forward_a_a_m_m_a_2_3_4_5_pool(x)  # 2022/2/23

        elif self.forward_choice == '':
            pass

        elif self.forward_choice == "m_5":
            return self._forward_m_5(x)

        elif self.forward_choice == "mul_add_5_output":
            return self._forward_mul_add_5_output(x)

        elif self.forward_choice == "mul_add_4_5_output":
            return self._forward_mul_add_4_5_output(x)

        elif self.forward_choice == "mul_add_3_4_5_output":
            return self._forward_mul_add_3_4_5_output(x)

        elif self.forward_choice == "m_a_m_a_m_2_3_4_5_pool":
            return self._foward_m_a_m_a_m_2_3_4_5_pool(x)

        elif self.forward_choice == "m_a_5":
            return self._forward_m_a_5(x)

        elif self.forward_choice == "m_a_4":
            return self._forward_m_a_4(x)

        elif self.forward_choice == "m_a_3":
            return self._forward_m_a_3(x)

        elif self.forward_choice == "m_a_2":
            return self._forward_m_a_2(x)

        else:
            raise NotImplementedError(f"Not Support this {self.forward_choice}")


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet18_GELU(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["act_layer"] = nn.GELU
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet18_LN(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["norm_layer"] = LayerNorm
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def implicit_resnet18_last_mul_add(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "last_mul_add"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def implicit_resnet18_mul_last_four_layer(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "mul_last_four_layer"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def implicit_resnet18_add_last_four_layer(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "add_last_four_layer"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def implicit_resnet18_add_all_layer(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "add_all_layer"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def implicit_resnet18_mul_add_every(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "mul_add_every"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_m_a_m_a_2_3_4_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "m_a_m_a_2_3_4_5"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_mul_add_5_output(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "mul_add_5_output"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_mul_add_4_5_output(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "mul_add_4_5_output"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_mul_add_3_4_5_output(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "mul_add_3_4_5_output"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_m_m_a_a_2_3_4_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "m_m_a_a_2_3_4_5"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_a_a_m_m_2_3_4_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "a_a_m_m_2_3_4_5"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_m_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "m_5"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_m_a_5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "m_a_5"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_m_a_4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "m_a_4"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_m_a_3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "m_a_3"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_m_a_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "m_a_2"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_m_m_a_a_2_3_4_5_GELU(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "m_m_a_a_2_3_4_5"
    kwargs["act_layer"] = nn.GELU
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_a_a_m_m_a_2_3_4_5_pool(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "a_a_m_m_a_2_3_4_5_pool"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_a_a_m_m_m_2_3_4_5_pool(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "a_a_m_m_m_2_3_4_5_pool"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def implicit_resnet18_m_a_m_a_m_2_3_4_5_pool(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs["forward_choice"] = "m_a_m_a_m_2_3_4_5_pool"
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
