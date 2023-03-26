import pytorch_lightning as pl
import torch
from torch import nn


class FingerprinterModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = FingerprinterEncoder()
        self.projection = FingerprinterProjection()
        self.loss = ContrastiveLoss()

    def forward(self, orig, rep):
        z_orig = self.projection(self.encoder(orig))
        z_rep = self.projection(self.encoder(rep))
        z = torch.cat([z_orig, z_rep], dim=1)
        return z

    def common_step(self, batch, batch_idx, mode="train"):
        orig, rep = batch
        z = self.forward(orig, rep)
        loss = self.loss(z)
        self.log(f"{mode}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, mode="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class FingerprinterEncoder(nn.Module):
    def __init__(self, dim=64, h=1024):
        super().__init__()
        self.conv1 = SpatiallySeparableConvBlock(1, dim, 3, 2, 256, 32)
        self.conv2 = SpatiallySeparableConvBlock(dim, dim, 3, 2, 128, 16)
        self.conv3 = SpatiallySeparableConvBlock(dim, 2 * dim, 3, 2, 64, 8)
        self.conv4 = SpatiallySeparableConvBlock(2 * dim, 2 * dim, 3, 2, 32, 4)
        self.conv5 = SpatiallySeparableConvBlock(2 * dim, 4 * dim, 3, 2, 16, 2)
        self.conv6 = SpatiallySeparableConvBlock(4 * dim, 4 * dim, 3, 2, 8, 1)
        self.conv7 = SpatiallySeparableConvBlock(4 * dim, h, 3, 2, 4, 1)
        self.conv8 = SpatiallySeparableConvBlock(h, h, 3, 2, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return x.squeeze(-1).squeeze(-1)


class FingerprinterProjection(nn.Module):
    def __init__(self, dim=64, h=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(h // dim, 32, 1)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(32, 1, 1)
        self.split_size = h // dim

    def forward(self, x):
        # split
        x = torch.stack(torch.split(x, self.split_size, dim=1), dim=2)
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.conv2(x)
        c = x.squeeze(1).squeeze(-1)
        g = torch.linalg.norm(c, dim=1, ord=2, keepdim=True)
        return g


class SpatiallySeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, f, t):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            (1, kernel_size),
            stride=(1, stride),
            padding=(0, kernel_size // 2),
        )
        self.layer_norm1 = nn.LayerNorm((out_channels, f, (t - 1) // stride + 1))
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            (kernel_size, 1),
            stride=(stride, 1),
            padding=(kernel_size // 2, 0),
        )
        self.layer_norm2 = nn.LayerNorm(
            (out_channels, (f - 1) // stride + 1, (t - 1) // stride + 1)
        )

    def forward(self, x):
        x = self.relu(self.layer_norm1(self.conv1(x)))
        x = self.relu(self.layer_norm2(self.conv2(x)))
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z):
        z = z / self.temperature
        a = torch.mm(z, z.T)
        l = -torch.log(torch.exp(a).sum(dim=1) / a.shape[1])
        return l.mean()
