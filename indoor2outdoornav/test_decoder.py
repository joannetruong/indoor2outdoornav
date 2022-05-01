import torch
import torch.nn.functional as F
from torch import nn


class Upsample(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.interpolate(x, size=self.output_size, mode="bilinear")


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1024)  # reshape to 32x32
        # self.decoder = F.interpolate()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, (3, 3), (1, 1)),  # 34x34
            Upsample((64, 64)),
            nn.ConvTranspose2d(32, 64, (3, 3), (1, 1)),  # 66x66
            Upsample((128, 128)),
            nn.ConvTranspose2d(64, 32, (3, 3), (1, 1)),  # 130x130
            Upsample((254, 254)),
            nn.ConvTranspose2d(32, 1, (3, 3), (1, 1)),  # 256x256
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, 1, 32, 32)  # NCHW
        return self.decoder(x)


d = Decoder(512)
a = torch.rand(1, 512)
b = d(a)
print(b.shape)
