import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


def similarity_loss(y, tau):
    a = torch.matmul(y, y.T)
    a /= tau
    Ls = []
    for i in range(y.shape[0]):
        nn_self = torch.cat([a[i, :i], a[i, i + 1 :]])
        softmax = torch.nn.functional.log_softmax(nn_self, dim=0)
        Ls.append(softmax[i if i % 2 == 0 else i - 1])

    Ls = torch.stack(Ls)
    loss = torch.sum(Ls) / -y.shape[0]
    return loss


class FingerprinterModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = FingerprinterEncoder()
        self.projection = FingerprinterProjection()
        self.loss = ContrastiveLoss()

    def forward(self, orig, rep):
        z_orig = self.projection(self.encoder(orig))
        z_rep = self.projection(self.encoder(rep))

        return z_orig, z_rep

    def common_step(self, batch, batch_idx, mode="train"):
        orig, rep = batch
        z_orig, z_rep = self.forward(orig, rep)
        # Interleave the two vectors
        z = []
        for i in range(z_orig.shape[0]):
            z.append(z_orig[i])
            z.append(z_rep[i])
        z = torch.stack(z, dim=0)
        loss = similarity_loss(z, 0.05)
        print("loss: ", loss)
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
    def __init__(self, dim=64, h=1024, u=32):
        super().__init__()
        self.conv1 = nn.Conv1d(h, dim * u, (1,), groups=dim)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv1d(dim * u, dim, (1,), groups=dim)
        self.split_size = h // dim

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.conv2(x)
        c = x.squeeze(1).squeeze(-1)

        g = torch.nn.functional.normalize(c, p=2.0)
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
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        pos = torch.mm(z1, z2.T)  # positive embedding
        cos = torch.mul(z1, z2)
        neg = torch.sum(cos, dim=1, keepdim=True)  # negative embedding
        mask = torch.eye(neg.shape[0]) * 1e12
        neg = neg - mask
        logits = torch.cat((pos, neg), dim=1)
        logits = logits / self.temperature
        loss_target = torch.zeros(logits.shape[0], dtype=torch.long)
        batch_loss = F.cross_entropy(logits, loss_target)
        return batch_loss

    # def forward(self, z):
    # a = torch.mm(z, z.T)
    # a = a / self.temperature
    # l = -torch.log(torch.exp(a).sum(dim=1) / a.shape[1])
    # num_pairs = int(z.shape[0] / 2)
    # l = []
    # for i in range(0, num_pairs, 2):
    #     p1 = z[i]
    #     p2 = z[i + 1]
    #     a = torch.mm(p1.view(1, -1), p2.view(-1, 1))
    #     a = a / self.temperature
    #     num = torch.exp(a)
    #     # g = a.fill_diagonal_(0)
    #     # denom = torch.exp(g).sum(dim=1)
    #     denom = torch.exp(a) * z.shape[0] - 1
    #     l.append(-torch.log(num / denom))
    # # e = torch.exp(a)
    # # numerator = e.sum(dim=1)
    # # d_e = e.fill_diagonal_(0)
    # # denominator = d_e.sum(dim=1)
    # # l = -torch.log(numerator / denominator)
    # # loss = 0
    # # for i in range(0, l.shape[0], 2):
    # #     loss += l[i]
    # return torch.sum(torch.stack(l)) / num_pairs
