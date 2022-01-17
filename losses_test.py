import unittest
import torch


class LossTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(LossTest, self).__init__(*args, **kwargs)
        self.batch_size = 10
        self.embedding = torch.randn(self.batch_size, 128)
        self.label = torch.arange(self.batch_size)

    def test_loss_by_atuhor(self):
        from losses import SupConLoss
        loss_fun = SupConLoss(temperature=0.1)
        embedding = torch.cat([self.embedding.unsqueeze(1), self.embedding.unsqueeze(1)], dim=1)
        label = torch.cat([self.label, self.label], dim=0)
        print(loss_fun(self.embedding, self.label))

    def test_loos_pytorch_metrics(self):
        from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
        loss_fun = SupConLoss(temperature=0.1)
        print(loss_fun(self.embedding, self.label))
