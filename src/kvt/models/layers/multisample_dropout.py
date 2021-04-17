import torch
from torch import nn


class MultiSampleDropout(nn.Module):
    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        super().__init__()
        self.num_samples = num_samples
        for i in range(num_samples):
            setattr(self, 'dropout{}'.format(i), nn.Dropout(dropout_rate))
            setattr(self, 'fc{}'.format(i), nn.Linear(in_features, out_features))
    
    def forward(self, x):
        logits = []
        for i in range(self.num_samples):
            dropout = getattr(self, 'dropout{}'.format(i))
            fc = getattr(self, 'fc{}'.format(i))
            x_ = dropout(x)
            x_ = fc(x_)
            logits.append(x_)
        return torch.stack(logits).mean(dim=0)
