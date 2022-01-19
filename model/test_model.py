import unittest
import torch
import torch.nn as nn
from .resnet_implict import implicit_resnet50, implicit_resnext50_32x4d, ImplicitMul, ImplicitAdd
from .clrnet import CLRBackbone, CLRHead
from .coatnet import *
from .convnext_clr import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResNetImplictTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ResNetImplictTest, self).__init__(*args, **kwargs)

    def test_resnet50implict(self):
        net = implicit_resnet50(pretrained=False)
        x = torch.randn(10, 3, 224, 224)
        y = net(x)
        print(y.shape)

    def test_resnext50implict(self):
        net = implicit_resnext50_32x4d(pretrained=False)
        x = torch.randn(10, 3, 224, 224)
        y = net(x)
        print(y.shape)


class ImplictLayerTest(unittest.TestCase):
    def test_add(self):
        x = torch.rand(10, 10, 2, 2)
        i_add = ImplicitAdd(channel=10)
        y = i_add(x)

    def test_mul(self):
        x = torch.rand(1, 10, 2, 2)
        i_mul = ImplicitMul(channel=10)
        y = i_mul(x)


class CLRBackboneTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CLRBackboneTest, self).__init__(*args, **kwargs)
        self.batch_size = 10
        self.dummy_image = torch.randn(self.batch_size, 3, 224, 224)
        self.feat_dim = 128

    def test_CLRBackbone_0(self):
        net = CLRBackbone(name='implicit_resnet50', head='mlp_bn', feat_dim=self.feat_dim)
        y = net(self.dummy_image)
        self.assertEqual(torch.Size([self.batch_size, self.feat_dim]), y.size())

    def test_CLRBackbone_1(self):
        net = CLRBackbone(name='implicit_resnext50', head='mlp_bn', feat_dim=self.feat_dim)
        y = net(self.dummy_image)
        self.assertEqual(torch.Size([self.batch_size, self.feat_dim]), y.size())

    def test_CLRBackbone_convnext_tiny(self):
        net = CLRBackbone(name='convnext_tiny', head='mlp_bn', feat_dim=self.feat_dim)
        y = net(self.dummy_image)
        self.assertEqual(torch.Size([self.batch_size, self.feat_dim]), y.size())

    def test_CLRBackbone_convnext_base(self):
        net = CLRBackbone(name='convnext_base', head='mlp_bn', feat_dim=self.feat_dim)
        y = net(self.dummy_image)
        self.assertEqual(torch.Size([self.batch_size, self.feat_dim]), y.size())


class CLRHeadTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CLRHeadTest, self).__init__(*args, **kwargs)
        self.dummy_present = torch.randn(10, 2048)

    def test_CLRHead_0(self):
        head = CLRHead('mlp')
        out = head(self.dummy_present)
        self.assertEqual(torch.Size([10, 128]), out.shape)

    def test_CLRHead_1(self):
        head = CLRHead('mlp_bn')
        out = head(self.dummy_present)
        self.assertEqual(torch.Size([10, 128]), out.shape)

    def test_CLRHead_2(self):
        head = CLRHead('2mlp_bn')
        out = head(self.dummy_present)
        self.assertEqual(torch.Size([10, 128]), out.shape)


class CoAtNetTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 10
        self.dummy_image = torch.randn(self.batch_size, 3, 224, 224)

    def test_coatnet1(self):
        net = coatnet_1()
        y = net(self.dummy_image)
        print(count_parameters(net))
        self.assertEqual(torch.Size([self.batch_size, 1000]), y.size())


class ConvnextTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 10
        self.dummy_image = torch.randn(self.batch_size, 3, 224, 224)

    def test_convnext_base(self):
        net = convnext_base(pretrained=False)
        output = net(self.dummy_image)
        print(count_parameters(net))
        print(output.shape)

    def test_convnext_tiny(self):
        net = convnext_tiny(pretrained=False)
        output = net(self.dummy_image)
        print(count_parameters(net))
        print(output.shape)

    def test_convnext_small(self):
        net = convnext_small(pretrained=False)
        output = net(self.dummy_image)
        print(count_parameters(net))
        print(output.shape)

    def test_convnext_large(self):
        net = convnext_large(pretrained=False)
        output = net(self.dummy_image)
        print(count_parameters(net))
        print(output.shape)
