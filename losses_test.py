import torch
from losses import SupConLoss
loss = SupConLoss(temperature=0.1)

batch_size = 10
features = torch.randn(batch_size, 2, 128)
labels = torch.randint(low=0, high=2, size=(batch_size,))
print(loss(features, labels))

from pytorch_metric_learning.losses import SupConLoss
loss = SupConLoss(temperature=0.1)
features = torch.cat(torch.unbind(features,dim=1),dim=0)
labels = labels.repeat(2)
print(loss(features, labels))