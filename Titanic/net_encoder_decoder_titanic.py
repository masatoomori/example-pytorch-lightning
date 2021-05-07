import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_dims, num_classes):
        super(Encoder, self).__init__()
        num_dim = 1
        for d in num_dims:
            num_dim *= d
        self.encoder = nn.Sequential(
            nn.Linear(num_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_dims, num_classes):
        super(Decoder, self).__init__()
        num_dim = 1
        for d in num_dims:
            num_dim *= d
        self.decoder = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, num_dim)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
