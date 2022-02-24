import unittest
import torch
import torch.nn as nn
from .resnet_implict import *
from .clrnet import CLRBackbone, CLRHead
from .coatnet import *
from .convnext_clr import *
from .cct import *
from .resnet18_sp import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CCTTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CCTTest, self).__init__(*args, **kwargs)
        self.input = torch.randn(2, 3, 224, 224)

    def test_cct(self):
        net = cct_2()
        y = net(self.input)
        print(y.size())


class ResNetImplictTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ResNetImplictTest, self).__init__(*args, **kwargs)
        self.batch_size = 10
        self.dummy_image = torch.randn(self.batch_size, 3, 224, 224)

    def test_resnet18_GELU(self):
        net = resnet18_GELU(pretrained=False)
        out = net(self.dummy_image)
        self.assertEqual(torch.Size([10, 512]), out.shape)

    def test_resnet18_LN(self):
        net = resnet18_LN(pretrained=False)
        out = net(self.dummy_image)
        self.assertEqual(torch.Size([10, 512]), out.shape)

    def test_implicit_resnet18_m_5(self):
        net = implicit_resnet18_m_5()
        out = net(self.dummy_image)
        self.assertEqual(torch.Size([10, 512]), out.shape)


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
        net = CLRBackbone(name='implicit_resnet18', head='mlp_bn', feat_dim=self.feat_dim)
        y = net(self.dummy_image)
        self.assertEqual(torch.Size([self.batch_size, self.feat_dim]), y.size())

    def test_CLRBackbone_convnext_tiny(self):
        net = CLRBackbone(name='convnext_tiny', head='mlp_bn', feat_dim=self.feat_dim)
        y = net(self.dummy_image)
        self.assertEqual(torch.Size([self.batch_size, self.feat_dim]), y.size())

    def test_CLRBackbone_convnext_base(self):
        net = CLRBackbone(name='coatnet_0', head='mlp_bn', feat_dim=self.feat_dim)
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


class CCTTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 10
        self.dummy_image = torch.randn(self.batch_size, 3, 224, 224)

    def test_cct_7_7x2_224(self):
        net = cct_7_7x2_224(num_classes=512)
        y = net(self.dummy_image)
        self.assertEqual(torch.Size([self.batch_size, 512]), y.size())

class ResNetTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 10
        self.dummy_image = torch.randn(self.batch_size, 3, 224, 224)

    def test_resnet18_sq(self):
        net = resnet18_sp(num_classes=512)
        y = net(self.dummy_image)
        self.assertEqual(torch.Size([self.batch_size, 512]), y.size())