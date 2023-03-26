import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pathlib import Path
import torchaudio
from torch.utils.data import random_split
import torch.nn.functional as F

sample_rate = 8000
chunk_size = 1.2 * sample_rate
clipped_chunk_size = 1.0 * sample_rate
length = 30
chunk_overlap = 0.5
chunks_per_song = length * sample_rate // chunk_overlap


class FingerprintDataset(Dataset):
    def __init__(self, data_dir, batch_size, anchors_per_batch=8):
        super().__init__()
        self.root = Path(data_dir)
        self.files = list(self.root.glob("*.wav"))
        shuffle_indices = torch.randperm(len(self.files))
        self.files = [self.files[i] for i in shuffle_indices]
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=256
        )

    def __len__(self):
        return len(self.files) * chunks_per_song

    def __getitem__(self, idx):
        song_id = idx // (chunks_per_song)
        chunk_id = idx % (chunks_per_song)
        data, sr = torchaudio.load(song_id)
        data = torchaudio.functional.resample(data, sr, sample_rate)
        chunk = data[
            :, chunk_id * chunk_overlap : (chunk_id * chunk_overlap) + chunk_size
        ]
        # Sum to mono
        if chunk.shape[0] > 1:
            chunk = torch.sum(chunk, dim=0, keepdim=True)

        # pick random 1s portion
        start1 = torch.randint(0, chunk.shape[1] - clipped_chunk_size, (1,))
        start2 = torch.randint(0, chunk.shape[1] - clipped_chunk_size, (1,))
        x_org = chunk[:, start1 : start1 + clipped_chunk_size]
        x_rep = chunk[:, start2 : start2 + clipped_chunk_size]
        # TD Augmentations
        # Background mixing
        # IR Filter
        # Spectrogram
        X_org = self.mel(x_org)
        X_rep = self.mel(x_rep)
        # Spectrogram Augmentations

        return X_org, X_rep


class FingerprintDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_split = int(len(self.dataset) * 0.75)
        val_split = len(self.dataset) - train_split
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_split, val_split]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
